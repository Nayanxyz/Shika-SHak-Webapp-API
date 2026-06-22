import os
import json
import logging
import asyncio
import time
import uuid
import hashlib
import re
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import redis.asyncio as redis
import jwt

from supabase import create_client, Client

from sympy import simplify
from sympy.parsing.latex import parse_latex
from sympy.core.expr import Expr

from groq import Groq

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 1. CONFIGURATION

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))

    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    ENV = os.getenv("ENV", "development")
    CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

    GAME_TIME_PER_QUESTION = int(os.getenv("GAME_TIME_PER_QUESTION", "60"))
    MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", "4"))

    # Session cleanup intervals
    PRACTICE_SESSION_TTL = int(os.getenv("PRACTICE_SESSION_TTL", "3600"))  # 1 hour
    ROOM_CLEANUP_INTERVAL = int(os.getenv("ROOM_CLEANUP_INTERVAL", "300"))  # 5 minutes
    ROOM_MAX_IDLE = int(os.getenv("ROOM_MAX_IDLE", "1800"))  # 30 minutes


REQUIRED = [
    ("GROQ_API_KEY", Config.GROQ_API_KEY),
    ("SUPABASE_URL", Config.SUPABASE_URL),
    ("SUPABASE_KEY", Config.SUPABASE_KEY),
    ("JWT_SECRET", Config.JWT_SECRET),
]

missing = [name for name, val in REQUIRED if not val]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

groq_client = Groq(api_key=Config.GROQ_API_KEY)
supabase_client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)


# 2. REDIS CACHE

class CacheManager:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._ttl = Config.REDIS_CACHE_TTL

    async def connect(self):
        try:
            self._redis = redis.from_url(Config.REDIS_URL, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis failed: {e}. Caching disabled.")
            self._redis = None

    async def disconnect(self):
        if self._redis:
            await self._redis.close()

    def _key(self, subject: str, difficulty: str, chapters: List[Dict]) -> str:
        chapters_str = json.dumps(chapters, sort_keys=True)
        hash_input = f"{subject}:{difficulty}:{chapters_str}"
        return f"exam:{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"

    async def get(self, subject: str, difficulty: str, chapters: List[Dict]) -> Optional[Dict]:
        if not self._redis:
            return None
        try:
            cached = await self._redis.get(self._key(subject, difficulty, chapters))
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(self, subject: str, difficulty: str, chapters: List[Dict], data: Dict):
        if not self._redis:
            return
        try:
            await self._redis.setex(self._key(subject, difficulty, chapters), self._ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    async def invalidate(self, pattern: str = "exam:*"):
        if not self._redis:
            return
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")


cache_manager = CacheManager()


# 3. MODELS

class Subject(str, Enum):
    MATH = "MATH"
    PHYSICS = "PHYSICS"
    CHEMISTRY = "CHEMISTRY"
    BIOLOGY = "BIOLOGY"


class Difficulty(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


class GameMode(str, Enum):
    PRACTICE = "PRACTICE"
    BATTLE = "BATTLE"


class GameStatus(str, Enum):
    WAITING = "WAITING"
    STARTING = "STARTING"
    PLAYING = "PLAYING"
    ANSWER_PHASE = "ANSWER_PHASE"
    LEADERBOARD = "LEADERBOARD"
    FINISHED = "FINISHED"


class ChapterMixItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=20)
    name: str = Field(..., min_length=1, max_length=200)


class ExamGenerationRequest(BaseModel):
    subject: Subject
    difficulty: Difficulty
    chapter_mix: List[ChapterMixItem] = Field(..., min_length=5, max_length=5)

    @field_validator("chapter_mix")
    @classmethod
    def validate_unique(cls, v):
        ids = [ch.id for ch in v]
        if len(set(ids)) != len(ids):
            raise ValueError("Chapter IDs must be unique")
        return v


class Option(BaseModel):
    id: str = Field(..., pattern="^[A-D]$")
    text: str = Field(..., min_length=1)


class Question(BaseModel):
    id: int
    chapter_id: str
    chapter_name: str
    question_text: str
    options: List[Option]
    correct_option_id: str = Field(..., pattern="^[A-D]$")
    explanation: str
    difficulty: Difficulty


class ExamResponse(BaseModel):
    subject: Subject
    difficulty: Difficulty
    questions: List[Question]
    generated_at: str
    cached: bool = False
    exam_id: Optional[str] = None


class PracticeAnswerRequest(BaseModel):
    session_id: str
    question_number: int = Field(..., ge=1, le=5)
    selected_option: Optional[str] = Field(None, pattern="^[A-D]$")


class PracticeResultsResponse(BaseModel):
    session_id: str
    total_score: int
    correct_count: int
    wrong_count: int
    accuracy: float
    completed: bool


# 4. GROQ JSON SCHEMA

QUESTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": ["A", "B", "C", "D"]},
                                "text": {"type": "string"}
                            },
                            "required": ["id", "text"]
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "correct_option_id": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "explanation": {"type": "string"}
                },
                "required": ["question_text", "options", "correct_option_id", "explanation"]
            },
            "minItems": 5,
            "maxItems": 5
        }
    },
    "required": ["questions"]
}


