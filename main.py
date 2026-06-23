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


# 5. AUTHENTICATION

security = HTTPBearer(auto_error=False)


class AuthManager:
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """
        Verify a Supabase JWT by checking the user with Supabase server-side.
        This handles both HS256 and ECC tokens.
        """
        try:
            # Use Supabase client to verify the token
            # Set the auth token temporarily
            supabase_client.auth.set_session(token, "")
            # Get the user - this validates the token with Supabase
            response = supabase_client.auth.get_user()
            user = response.user

            if not user:
                logger.warning("Supabase auth returned no user")
                return None

            return {
                "sub": user.id,
                "email": user.email,
                "role": "authenticated"
            }
        except Exception as e:
            logger.warning(f"Supabase token verification failed: {e}")
            return None

    @classmethod
    async def get_current_user(cls, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        payload = cls.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return {"user_id": payload["sub"], "email": payload["email"]}

# 6. RATE LIMITING

class RateLimiter:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def connect(self):
        try:
            self._redis = redis.from_url(Config.REDIS_URL, decode_responses=True)
            await self._redis.ping()
            logger.info("Rate limiter Redis connected")
        except Exception as e:
            logger.warning(f"Rate limiter Redis failed: {e}")
            self._redis = None

    async def disconnect(self):
        if self._redis:
            await self._redis.close()

    async def is_allowed(self, key: str) -> bool:
        if not self._redis:
            return True
        try:
            now = time.time()
            window_start = now - Config.RATE_LIMIT_WINDOW
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            pipe.zcard(f"rate_limit:{key}")
            _, current = await pipe.execute()

            if current >= Config.RATE_LIMIT_REQUESTS:
                return False

            pipe = self._redis.pipeline()
            pipe.zadd(f"rate_limit:{key}", {str(now): now})
            pipe.expire(f"rate_limit:{key}", Config.RATE_LIMIT_WINDOW)
            await pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Rate limit check error: {e}")
            return True


rate_limiter = RateLimiter()


async def rate_limit_dependency(request: Request, user: Dict = Depends(AuthManager.get_current_user)):
    if not await rate_limiter.is_allowed(user["user_id"]):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {Config.RATE_LIMIT_REQUESTS} req/{Config.RATE_LIMIT_WINDOW}s"
        )
    return user


# 7. SUPABASE SERVICE (Minimal - metadata only)

class SupabaseService:
    @staticmethod
    async def store_session(user_id: str, subject: str, difficulty: str,
                           chapter_ids: List[str], mode: str, score: int = 0,
                           correct_count: int = 0, wrong_count: int = 0,
                           unanswered_count: int = 0, accuracy: float = 0.0,
                           total_time_ms: int = 0) -> Optional[str]:
        try:
            result = supabase_client.table("sessions").insert({
                "user_id": user_id,
                "subject": subject,
                "difficulty": difficulty,
                "chapter_ids": chapter_ids,
                "mode": mode,
                "score": score,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "unanswered_count": unanswered_count,
                "accuracy": accuracy,
                "total_time_ms": total_time_ms,
                "completed": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Supabase store_session error: {e}")
            return None

    @staticmethod
    async def get_session_history(user_id: str, limit: int = 20, offset: int = 0):
        try:
            result = supabase_client.table("sessions")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            return {"sessions": result.data, "total": len(result.data)}
        except Exception as e:
            logger.error(f"History error: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch history")


# 8. VALIDATORS

class MathPhysicsValidator:
    SKIP_TOKENS = ["\\lim", "\\matrix", "\\begin", "\\cases", "\\sum", "\\int", "\\prod"]

    def _clean(self, text: str) -> str:
        return text.replace("$", "").strip()

    def _should_skip(self, latex: str) -> bool:
        return any(t in latex for t in self.SKIP_TOKENS)

    def _parse(self, latex: str) -> Optional[Expr]:
        try:
            return parse_latex(latex)
        except Exception:
            return None

    def validate(self, questions: List[Dict]) -> bool:
        for q_idx, q in enumerate(questions):
            options = q.get("options", [])
            if len(options) != 4:
                logger.error(f"[Q{q_idx+1}] Expected 4 options, got {len(options)}")
                return False

            texts = [self._clean(opt["text"]) for opt in options]
            if len(set(texts)) != len(options):
                logger.error(f"[Q{q_idx+1}] Duplicate option text detected")
                return False

            parsed = []
            for opt in options:
                clean = self._clean(opt["text"])
                if self._should_skip(clean):
                    continue
                expr = self._parse(clean)
                if expr is not None:
                    parsed.append((opt["id"], expr))

            for i in range(len(parsed)):
                for j in range(i + 1, len(parsed)):
                    id1, e1 = parsed[i]
                    id2, e2 = parsed[j]
                    try:
                        if simplify(e1 - e2) == 0 or simplify(e1).equals(simplify(e2)):
                            logger.error(f"[Q{q_idx+1}] Identical options: {id1} and {id2}")
                            return False
                    except Exception:
                        continue
        return True


class ChemistryValidator:
    def validate(self, questions: List[Dict]) -> bool:
        for q_idx, q in enumerate(questions):
            options = q.get("options", [])
            if len(options) != 4:
                logger.error(f"[Q{q_idx+1}] Expected 4 options, got {len(options)}")
                return False
            texts = [opt["text"].strip() for opt in options]
            if len(set(texts)) != len(options):
                logger.error(f"[Q{q_idx+1}] Duplicate option text detected")
                return False
        return True


class BiologyValidator:
    def validate(self, questions: List[Dict]) -> bool:
        for q_idx, q in enumerate(questions):
            options = q.get("options", [])
            if len(options) != 4:
                logger.error(f"[Q{q_idx+1}] Expected 4 options, got {len(options)}")
                return False
            texts = [opt["text"].strip() for opt in options]
            if len(set(texts)) != len(options):
                logger.error(f"[Q{q_idx+1}] Duplicate option text detected")
                return False
        return True


class ValidationRouter:
    def __init__(self):
        self.math_physics = MathPhysicsValidator()
        self.chemistry = ChemistryValidator()
        self.biology = BiologyValidator()

    def route_and_verify(self, subject: Subject, questions: List[Dict]) -> bool:
        if subject in [Subject.MATH, Subject.PHYSICS]:
            return self.math_physics.validate(questions)
        elif subject == Subject.CHEMISTRY:
            return self.chemistry.validate(questions)
        elif subject == Subject.BIOLOGY:
            return self.biology.validate(questions)
        return False


# 9. GROQ QUESTION GENERATION SERVICE

class QuestionService:
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self):
        self.validator = ValidationRouter()

    def _build_system_prompt(self, subject: Subject, difficulty: Difficulty) -> str:
        subjects = {
            Subject.MATH: "Expert JEE Mathematics professor",
            Subject.PHYSICS: "Expert JEE Physics professor",
            Subject.CHEMISTRY: "Expert JEE/NEET Chemistry professor",
            Subject.BIOLOGY: "Expert NEET Biology professor",
        }
        diffs = {
            Difficulty.LOW: "Foundation level. Direct application. 2-3 min solve time.",
            Difficulty.HIGH: "Advanced competitive. Multi-step reasoning. 5-8 min solve time.",
        }
        return f"""You are a {subjects[subject]} creating competitive exam questions.

{diffs[difficulty]}

CRITICAL RULES:
1. Generate EXACTLY 5 questions, one per chapter.
2. Each question has 4 options: A, B, C, D.
3. Options must be DISTINCT — no duplicates.
4. Use LaTeX ($...$) for math/chemical notation. ALWAYS use double backslashes for LaTeX commands: \\frac, \\sum, \\alpha, \\beta, \\pi, etc.
5. Correct answer must be unambiguous.
6. Include detailed explanation with proper LaTeX.
7. NEVER use single backslash in LaTeX commands — always double backslash.

You MUST respond with valid JSON only. No markdown, no extra text.
CRITICAL: Respond with ONLY valid JSON. No explanations outside JSON. No thinking process. No "let me calculate". Just the JSON object with the "questions" array.
"""

    def _build_user_prompt(self, subject: Subject, difficulty: Difficulty, chapters: List[ChapterMixItem]) -> str:
        chapters_text = "\n".join(f"{i+1}. [{ch.id}] {ch.name}" for i, ch in enumerate(chapters))
        schema_hint = json.dumps(QUESTION_JSON_SCHEMA, indent=2)
        return f"""Generate 5 MCQ questions for {subject.value} ({difficulty.value}).

CHAPTERS:
{chapters_text}

Respond with JSON matching this schema:
{schema_hint}

Remember: ONLY JSON. No other text. Use \\frac, \\sum, \\alpha etc. (double backslash) for LaTeX."""

