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

