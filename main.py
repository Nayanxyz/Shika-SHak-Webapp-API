import os
import json
import logging
import asyncio
import time
import uuid
import hashlib
import re
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, HTTPException, Depends, Request, Body, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import redis.asyncio as redis
import random
import jwt

from supabase import create_client, Client

try:
    from sympy import simplify
    from sympy.parsing.latex import parse_latex
    from sympy.core.expr import Expr
    SYMPY_LATEX_AVAILABLE = True
except ImportError:
    SYMPY_LATEX_AVAILABLE = False
    simplify = None  # type: ignore
    parse_latex = None  # type: ignore
    Expr = None  # type: ignore

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
    GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP")
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

    PRACTICE_SESSION_TTL = int(os.getenv("PRACTICE_SESSION_TTL", "3600"))
    ROOM_CLEANUP_INTERVAL = int(os.getenv("ROOM_CLEANUP_INTERVAL", "300"))
    ROOM_MAX_IDLE = int(os.getenv("ROOM_MAX_IDLE", "1800"))

REQUIRED = [
    ("GROQ_API_KEY", Config.GROQ_API_KEY),
    ("SUPABASE_URL", Config.SUPABASE_URL),
    ("SUPABASE_KEY", Config.SUPABASE_KEY),
    ("JWT_SECRET", Config.JWT_SECRET),
]

missing = [name for name, val in REQUIRED if not val]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

groq_primary = Groq(api_key=Config.GROQ_API_KEY, max_retries=0)
groq_backup = Groq(api_key=Config.GROQ_API_KEY_BACKUP, max_retries=0) if Config.GROQ_API_KEY_BACKUP else None

supabase_client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

_shared_redis: Optional[redis.Redis] = None

async def get_redis() -> redis.Redis:
    global _shared_redis
    if _shared_redis is None:
        _shared_redis = redis.from_url(Config.REDIS_URL, decode_responses=True)
        await _shared_redis.ping()
    return _shared_redis

# 2. REDIS CACHE

class CacheManager:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._ttl = Config.REDIS_CACHE_TTL

    async def connect(self):
        try:
            self._redis = await get_redis()
            logger.info("Redis connected (shared)")
        except Exception as e:
            logger.warning(f"Redis failed: {e}. Caching disabled.")
            self._redis = None

    async def disconnect(self):
        pass

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
    def create_token(user_id: str, email: str) -> str:
        payload = {
            "sub": user_id,
            "email": email,
            "exp": datetime.now(timezone.utc).timestamp() + Config.JWT_EXPIRE_MINUTES * 60,
            "iat": datetime.now(timezone.utc).timestamp(),
        }
        return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:

        try:
            response = supabase_client.auth.get_user(token)
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
            self._redis = await get_redis()
            logger.info("Rate limiter Redis connected (shared)")
        except Exception as e:
            logger.warning(f"Rate limiter Redis failed: {e}")
            self._redis = None

    async def disconnect(self):
        pass

    async def is_allowed(self, key: str) -> bool:
        if not self._redis:
            return True
        try:
            now = time.time()
            window_start = now - Config.RATE_LIMIT_WINDOW
            pipe = self._redis.pipeline()
            await pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            await pipe.zcard(f"rate_limit:{key}")
            _, current = await pipe.execute()

            if current >= Config.RATE_LIMIT_REQUESTS:
                return False

            pipe = self._redis.pipeline()
            await pipe.zadd(f"rate_limit:{key}", {str(now): now})
            await pipe.expire(f"rate_limit:{key}", Config.RATE_LIMIT_WINDOW)
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


# 7. VALIDATORS

class MathPhysicsValidator:
    SKIP_TOKENS = ["\\lim", "\\matrix", "\\begin", "\\cases", "\\sum", "\\int", "\\prod"]

    def _clean(self, text: str) -> str:
        return text.replace("$", "").strip()

    def _should_skip(self, latex: str) -> bool:
        return any(t in latex for t in self.SKIP_TOKENS)

    def _parse(self, latex: str) -> Optional[Any]:
        if not SYMPY_LATEX_AVAILABLE:
            return None
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

# 8. GROQ QUESTION GENERATION SERVICE

class QuestionService:
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self):
        self.validator = ValidationRouter()

    def _build_system_prompt(self, subject: Subject, difficulty: Difficulty, expected_count: int) -> str:
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
        1. Generate EXACTLY {expected_count} questions, one per chapter. NO MORE, NO LESS.
        2. Each question MUST have 4 options: A, B, C, D.
        3. Options must be COMPLETELY DIFFERENT — no duplicates, no similar text.
        4. Use LaTeX ($...$) for math/chemical notation.
        5. Correct answer must be unambiguous.
        6. Include detailed explanation with proper LaTeX.
        7. NEVER write LaTeX commands without the backslash. Always use \\command.
        8. In JSON, backslashes must be escaped as double backslash.

        YOU MUST respond with valid JSON only. The JSON must contain exactly 5 questions.
        """

    def _build_user_prompt(self, subject: Subject, difficulty: Difficulty, chapters: List[ChapterMixItem], expected_count: int) -> str:
        chapters_text = "\n".join(f"{i + 1}. [{ch.id}] {ch.name}" for i, ch in enumerate(chapters))
        schema_hint = json.dumps(QUESTION_JSON_SCHEMA, indent=2)
        seed = random.randint(1000, 999999)

        return f"""Generate 5 MCQ questions for {subject.value} ({difficulty.value}).

    SEED: {seed}

    CHAPTERS:
    {chapters_text}

    Respond with JSON matching this schema:
    {schema_hint}

    Remember: ONLY JSON. No other text. Use proper LaTeX inside $...$ like $\\frac{{a}}{{b}}$, $\\sqrt{{x}}$, $\\vec{{a}}$, $\\alpha$, $\\cdot$, etc.
    In JSON, write LaTeX commands with double backslash: \\\\frac, \\\\sqrt, \\\\cdot, \\\\alpha, etc.
    """

    def _sanitize_latex(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("\x0c", "\f")
        return text

    def _parse_response(self, content: str) -> Optional[List[Dict]]:
        if not content:
            return None

        try:
            data = json.loads(content)
            questions = data.get("questions", [])
            for q in questions:
                q["question_text"] = self._sanitize_latex(q.get("question_text", ""))
                q["explanation"] = self._sanitize_latex(q.get("explanation", ""))
                for opt in q.get("options", []):
                    opt["text"] = self._sanitize_latex(opt.get("text", ""))
            return questions
        except json.JSONDecodeError:
            pass

        try:
            match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                return data.get("questions", [])
        except Exception:
            pass

        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("questions", [])
        except Exception:
            pass

        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass

        return None

    def _validate_questions(self, questions: List[Dict], expected_count: int) -> bool:
        if not questions or len(questions) != expected_count:
            logger.warning(f"Expected {expected_count} questions, got {len(questions) if questions else 0}")
            return False
        for q_idx, q in enumerate(questions):
            required_keys = ["question_text", "options", "correct_option_id", "explanation"]
            missing_keys = [k for k in required_keys if k not in q]
            if missing_keys:
                logger.warning(f"[Q{q_idx+1}] Missing keys: {missing_keys}")
                return False
            if len(q.get("options", [])) != 4:
                logger.warning(f"[Q{q_idx+1}] Expected 4 options, got {len(q.get('options', []))}")
                return False
            opt_ids = [opt["id"] for opt in q["options"]]
            if set(opt_ids) != {"A", "B", "C", "D"}:
                logger.warning(f"[Q{q_idx+1}] Invalid option IDs: {opt_ids}")
                return False
            if q["correct_option_id"] not in {"A", "B", "C", "D"}:
                logger.warning(f"[Q{q_idx+1}] Invalid correct_option_id: {q['correct_option_id']}")
                return False
        return True

    async def generate(self, subject: Subject, difficulty: Difficulty, chapters: List[ChapterMixItem]) -> List[Dict]:
        expected_count = len(chapters)
        clients = [(groq_primary, "primary")]
        if groq_backup:
            clients.append((groq_backup, "backup"))

        for client, name in clients:
            for attempt in range(1, self.MAX_RETRIES + 1):
                logger.info(f"Groq {name} attempt {attempt}/{self.MAX_RETRIES}")

                try:
                    completion = client.chat.completions.create(
                        model=Config.GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": self._build_system_prompt(subject, difficulty, expected_count)},
                            {"role": "user", "content": self._build_user_prompt(subject, difficulty, chapters, expected_count)},
                        ],
                        temperature=0.9,
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                    )

                    content = completion.choices[0].message.content
                    if not content:
                        logger.warning(f"Empty response from Groq {name}")
                        continue

                    questions = self._parse_response(content)

                    if not questions:
                        logger.warning(f"Failed to parse questions from {name}")
                        continue

                    if not self._validate_questions(questions, expected_count):
                        logger.warning(f"Validation failed on {name}: {len(questions)} questions")
                        continue

                    if self.validator.route_and_verify(subject, questions):
                        logger.info(f"Questions generated via {name}")
                        return questions
                    else:
                        logger.warning(f"Subject validation failed on {name}, retrying...")

                except Exception as e:
                    logger.error(f"Groq {name} error: {e}")
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY * attempt)

            logger.warning(f"Groq {name} failed all retries, trying next...")

        raise HTTPException(status_code=422, detail="All Groq providers failed after retries")

# 9. GAME STATE (In-Memory with Cleanup)

@dataclass
class Player:
    sid: str
    user_id: str
    name: str
    is_host: bool = False
    is_connected: bool = True
    total_score: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    unanswered_count: int = 0
    current_streak: int = 0
    max_streak: int = 0
    total_time_ms: int = 0
    answers: Dict[int, Dict] = field(default_factory=dict)
    joined_at: float = field(default_factory=time.time)

@dataclass
class GameRoom:
    room_code: str
    mode: GameMode
    subject: Subject
    difficulty: Difficulty
    chapters: List[ChapterMixItem]
    questions: List[dict] = field(default_factory=list)
    players: Dict[str, Player] = field(default_factory=dict)
    status: GameStatus = GameStatus.WAITING
    max_players: int = 4
    time_per_question: int = 60
    current_question_index: int = -1
    question_start_time: Optional[float] = None
    question_timer_task: Optional[asyncio.Task] = None
    final_rankings: List[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, GameRoom] = {}
        self.player_rooms: Dict[str, str] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup(self):
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(Config.ROOM_CLEANUP_INTERVAL)
            await self._cleanup_old_rooms()

    async def _cleanup_old_rooms(self):
        now = time.time()
        to_remove = []
        for code, room in self.rooms.items():
            if room.status == GameStatus.FINISHED and now - room.last_activity > 600:
                to_remove.append(code)
            elif room.status == GameStatus.WAITING and now - room.created_at > Config.ROOM_MAX_IDLE:
                to_remove.append(code)

        for code in to_remove:
            room = self.rooms.pop(code, None)
            if room:
                for sid in list(room.players.keys()):
                    self.player_rooms.pop(sid, None)
                logger.info(f"Cleaned up room {code}")

    def generate_code(self) -> str:
        import random, string
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in self.rooms:
                return code

    def create(self, subject: Subject, difficulty: Difficulty, chapters: List[ChapterMixItem],
               host_sid: str, user_id: str, name: str, max_players: int = 4, time_per_q: int = 60) -> GameRoom:
        code = self.generate_code()
        room = GameRoom(
            room_code=code, mode=GameMode.BATTLE, subject=subject,
            difficulty=difficulty, chapters=chapters,
            max_players=max_players, time_per_question=time_per_q
        )
        room.players[host_sid] = Player(sid=host_sid, user_id=user_id, name=name, is_host=True)
        self.player_rooms[host_sid] = code
        self.rooms[code] = room
        self.start_cleanup()
        return room

    def join(self, code: str, sid: str, user_id: str, name: str) -> Optional[GameRoom]:
        room = self.rooms.get(code.upper())
        if not room or room.status != GameStatus.WAITING:
            return None
        if len(room.players) >= room.max_players:
            return None
        room.players[sid] = Player(sid=sid, user_id=user_id, name=name)
        room.last_activity = time.time()
        self.player_rooms[sid] = code
        return room

    def leave(self, sid: str) -> Optional[GameRoom]:
        code = self.player_rooms.pop(sid, None)
        if not code:
            return None
        room = self.rooms.get(code)
        if room:
            if sid in room.players:
                room.players[sid].is_connected = False
                room.last_activity = time.time()
            if room.players.get(sid, Player("", "", "")).is_host and room.status == GameStatus.WAITING:
                connected = [p for p in room.players.values() if p.is_connected and p.sid != sid]
                if connected:
                    connected[0].is_host = True
                else:
                    room.status = GameStatus.FINISHED
        return room

    def remove_player(self, sid: str) -> Optional[GameRoom]:
        code = self.player_rooms.get(sid)
        if not code:
            return None
        room = self.rooms.get(code)
        if room and sid in room.players:
            del room.players[sid]
            self.player_rooms.pop(sid, None)
            room.last_activity = time.time()
            if not room.players:
                self.rooms.pop(code, None)
            return room
        return None

    def get(self, code: str) -> Optional[GameRoom]:
        return self.rooms.get(code.upper())

    def get_by_sid(self, sid: str) -> Optional[GameRoom]:
        code = self.player_rooms.get(sid)
        return self.rooms.get(code) if code else None

    def get_room_info(self, code: str) -> Optional[Dict]:
        room = self.rooms.get(code.upper())
        if not room:
            return None
        return {
            "room_code": room.room_code,
            "status": room.status.value,
            "subject": room.subject.value,
            "difficulty": room.difficulty.value,
            "player_count": len(room.players),
            "max_players": room.max_players,
            "players": [
                {
                    "sid": p.sid,
                    "name": p.name,
                    "is_host": p.is_host,
                    "is_connected": p.is_connected,
                }
                for p in room.players.values()
            ],
        }

room_manager = RoomManager()

