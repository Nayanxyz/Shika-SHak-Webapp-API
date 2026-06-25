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
from groq import Groq
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
from sympy import simplify
from sympy.parsing.latex import parse_latex

SYMPY_LATEX_AVAILABLE = True


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

                    if simplify(e1 - e2) == 0 or simplify(e1).equals(simplify(e2)):
                        logger.error(f"[Q{q_idx+1}] Identical options: {id1} and {id2}")
                        return False

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

# 10. SCORING ENGINE

class ScoringEngine:
    CORRECT = 4
    WRONG = -1
    UNANSWERED = 0
    BONUS_50 = 0
    BONUS_25 = 0
    STREAK_3 = 0
    STREAK_5 = 0

    @classmethod
    def calculate(cls, selected: Optional[str], correct: str, time_taken_ms: int, total_time_ms: int, streak: int):
        is_correct = selected == correct if selected else False
        time_ratio = time_taken_ms / total_time_ms if total_time_ms > 0 else 1.0

        base = cls.CORRECT if is_correct else (cls.WRONG if selected else cls.UNANSWERED)
        time_bonus = 0
        streak_bonus = 0
        new_streak = streak + 1 if is_correct else 0

        if is_correct:
            if time_ratio <= 0.25:
                time_bonus = cls.BONUS_25
            elif time_ratio <= 0.50:
                time_bonus = cls.BONUS_50
            if new_streak >= 5:
                streak_bonus = cls.STREAK_5
            elif new_streak >= 3:
                streak_bonus = cls.STREAK_3

        return {
            "is_correct": is_correct,
            "base_score": base,
            "time_bonus": time_bonus,
            "streak_bonus": streak_bonus,
            "total_score": base + time_bonus + streak_bonus,
            "new_streak": new_streak,
        }

    @classmethod
    def rankings(cls, players: Dict[str, Player]):
        plist = sorted(players.values(), key=lambda p: (
            -p.total_score,
            -(p.correct_count / max(p.correct_count + p.wrong_count + p.unanswered_count, 1)),
            p.total_time_ms / max(p.correct_count + p.wrong_count, 1),
            -p.max_streak,
        ))
        return [{
            "rank": i + 1,
            "user_id": p.user_id,
            "name": p.name,
            "total_score": p.total_score,
            "correct_count": p.correct_count,
            "wrong_count": p.wrong_count,
            "unanswered_count": p.unanswered_count,
            "accuracy": round(p.correct_count / max(p.correct_count + p.wrong_count + p.unanswered_count, 1) * 100, 1),
            "max_streak": p.max_streak,
        } for i, p in enumerate(plist)]

# 11. SOCKET.IO EVENTS

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="http://localhost:3001",
    logger=True,
    engineio_logger=True,
)
socket_app = socketio.ASGIApp(sio)

def normalize_payload(data: Any) -> dict:
    if isinstance(data, str):
        try: return json.loads(data)
        except Exception: pass
    if isinstance(data, dict): return data
    return {}

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    await sio.emit("connected", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid):
    room = room_manager.get_by_sid(sid)
    if not room:
        return

    # 1. Update player status
    room_manager.leave(sid)

    # 2. Check if we should forfeit or continue
    active_players = [p for p in room.players.values() if p.is_connected]

    if len(active_players) < 2:
        # Not enough players left to play
        await sio.emit("room_forfeited", {"message": "Not enough players to continue. Match ended."},
                       room=room.room_code)
        room_manager.rooms.pop(room.room_code, None)
    else:
        # Notify remaining players someone left
        await sio.emit("player_left_notification", {"message": "A player has left the battle. Continuing..."},
                       room=room.room_code)
        # Update player list for UI
        await sio.emit("update_players", {
            "players": [{"sid": p.sid, "name": p.name, "is_host": p.is_host} for p in active_players]
        }, room=room.room_code)

@sio.on("create_room")
async def on_create(sid, data):
    try:
        payload = normalize_payload(data)
        all_chapters = [ChapterMixItem(**c) for c in payload["chapter_mix"]]

        room = room_manager.create(
            Subject(payload["subject"]),
            Difficulty(payload["difficulty"]),
            all_chapters,
            sid,
            payload.get("user_id", f"user_{sid[:8]}"),
            payload.get("player_name", "Player"),
            min(int(payload.get("max_players", 4)), 8),
            min(int(payload.get("time_per_question", 60)), 300)
        )

        await sio.enter_room(sid, room.room_code)

        await sio.emit("room_created", {
            "room_code": room.room_code,
            "mode": room.mode.value,
            "subject": room.subject.value,
            "difficulty": room.difficulty.value,
            "players": [{"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected} for p in
                        room.players.values()],
            "is_host": True
        }, to=sid)

        service = QuestionService()
        room.questions = await service.generate(room.subject, room.difficulty, all_chapters)

        for idx, q in enumerate(room.questions):
            q["question_number"] = idx + 1

        await sio.emit("questions_ready", {"ready": True}, room=room.room_code)

    except Exception as e:
        logger.error(f"WebSocket Room Configuration Critical Crash: {e}", exc_info=True)
        await sio.emit("error", {"message": f"Initialization runtime error: {e}"}, to=sid)
        await sio.emit("generation_failed", {"message": "AI overloaded, please try again."}, to=sid)


@sio.on("join_room")
async def on_join(sid, data):
    try:
        if not isinstance(data, dict) or "room_code" not in data:
            await sio.emit("error", {"message": "Room code required"}, to=sid)
            return

        user_id = data.get("user_id", f"user_{sid[:8]}")
        room = room_manager.join(
            data["room_code"], sid,
            user_id,
            data.get("player_name", "Player")
        )
        if not room:
            await sio.emit("error", {"message": "Room not found or full"}, to=sid)
            return

        await sio.enter_room(sid, room.room_code)
        await sio.emit("room_joined", {
            "room_code": room.room_code,
            "players": [
                {"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected}
                for p in room.players.values()
            ],
            "is_host": False,
        }, to=sid)

        await sio.emit("player_joined", {
            "players": [
                {"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected}
                for p in room.players.values()
            ]
        }, room=room.room_code)
    except Exception as e:
        logger.error(f"Join room error: {e}")
        await sio.emit("error", {"message": "Failed to join room"}, to=sid)

@sio.on("rejoin_room")
async def on_rejoin(sid, data):
    try:
        room_code = data.get("room_code")
        user_id = data.get("user_id")
        if not room_code or not user_id:
            await sio.emit("error", {"message": "Room code and user_id required"}, to=sid)
            return

        room = room_manager.get(room_code)
        if not room:
            await sio.emit("error", {"message": "Room not found"}, to=sid)
            return

        old_player = next((p for p in room.players.values() if p.user_id == user_id), None)
        if old_player:
            room_manager.player_rooms.pop(old_player.sid, None)
            old_player.sid = sid
            old_player.is_connected = True
            room_manager.player_rooms[sid] = room_code
            await sio.enter_room(sid, room_code)

            await sio.emit("room_joined", {
                "room_code": room.room_code,
                "players": [{"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected} for p in room.players.values()],
                "is_host": old_player.is_host,
            }, to=sid)

            await sio.emit("player_joined", {
                "players": [{"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected} for p in room.players.values()]
            }, room=room.room_code)

            if room.status in (GameStatus.PLAYING, GameStatus.ANSWER_PHASE, GameStatus.LEADERBOARD) and room.current_question_index >= 0:
                q = room.questions[room.current_question_index]
                await sio.emit("question_start", {
                    "question_number": room.current_question_index + 1,
                    "total_questions": len(room.questions),
                    "question_text": q["question_text"],
                    "options": q["options"],
                    "time_limit": room.time_per_question,
                }, to=sid)
        else:
            await sio.emit("error", {"message": "Player not found in room"}, to=sid)
    except Exception as e:
        logger.error(f"Rejoin error: {e}")
        await sio.emit("error", {"message": "Failed to rejoin"}, to=sid)

@sio.on("leave_room")
async def on_leave(sid, data):
    try:
        room = room_manager.leave(sid)
        if room:
            # If they leave during an active battle or leaderboard
            if room.status in [GameStatus.STARTING, GameStatus.PLAYING, GameStatus.ANSWER_PHASE, GameStatus.LEADERBOARD,
                               GameStatus.FINISHED]:
                await sio.emit("room_forfeited", {"message": "A player disconnected. Match forfeited."},
                               room=room.room_code)
                # Nuke the room from memory completely
                room_manager.rooms.pop(room.room_code, None)
            else:
                # Normal lobby leave
                await sio.emit("player_left", {
                    "sid": sid,
                    "players": [{"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected} for
                                p in room.players.values()]
                }, room=room.room_code)
    except Exception as e:
        logger.error(f"Leave room error: {e}")
        await sio.emit("error", {"message": "Failed to leave room"}, to=sid)

@sio.on("kick_player")
async def on_kick(sid, data):
    try:
        room = room_manager.get_by_sid(sid)
        if not room or not room.players.get(sid, Player("", "", "")).is_host:
            await sio.emit("error", {"message": "Only host can kick players"}, to=sid)
            return

        target_sid = data.get("target_sid")
        if not target_sid or target_sid not in room.players:
            await sio.emit("error", {"message": "Player not found"}, to=sid)
            return

        if target_sid == sid:
            await sio.emit("error", {"message": "Cannot kick yourself"}, to=sid)
            return

        room_manager.remove_player(target_sid)
        await sio.leave_room(target_sid, room.room_code)

        await sio.emit("kicked", {"room_code": room.room_code}, to=target_sid)
        await sio.emit("player_kicked", {
            "sid": target_sid,
            "players": [
                {"sid": p.sid, "name": p.name, "is_host": p.is_host, "is_connected": p.is_connected}
                for p in room.players.values()
            ]
        }, room=room.room_code)
    except Exception as e:
        logger.error(f"Kick player error: {e}")
        await sio.emit("error", {"message": "Failed to kick player"}, to=sid)

@sio.on("get_room_info")
async def on_get_room_info(sid, data):
    try:
        room = room_manager.get_by_sid(sid)
        if not room:
            await sio.emit("error", {"message": "Not in a room"}, to=sid)
            return

        info = room_manager.get_room_info(room.room_code)
        await sio.emit("room_info", info, to=sid)
    except Exception as e:
        logger.error(f"Get room info error: {e}")


@sio.on("start_game")
async def on_start(sid, data):
    room = room_manager.get_by_sid(sid)
    if not room or not room.players.get(sid, Player("", "", "")).is_host or not room.questions or room.status != GameStatus.WAITING:
        return
    room.status = GameStatus.STARTING

    # 1. Emit the signal to force the React router to change pages
    await sio.emit("game_starting", {"room_code": room.room_code}, room=room.room_code)

    # 2. WAIT for 1.5 seconds so the browser can actually load the BattleGame screen
    await asyncio.sleep(1.5)

    # 3. Now drop the first question
    await start_question(room, 0)

async def start_question(room: GameRoom, idx: int):
    if room.question_timer_task and not room.question_timer_task.done():
        room.question_timer_task.cancel()
        try:
            await room.question_timer_task
        except asyncio.CancelledError:
            pass

    room.current_question_index = idx
    room.status = GameStatus.PLAYING
    room.question_start_time = time.time()
    room.last_activity = time.time()
    q = room.questions[idx]

    await sio.emit("question_start", {
        "question_number": idx + 1,
        "total_questions": len(room.questions),
        "question_text": q["question_text"],
        "options": q["options"],
        "time_limit": room.time_per_question,
    }, room=room.room_code)

    room.question_timer_task = asyncio.create_task(question_timer(room, idx))

async def question_timer(room: GameRoom, idx: int):
    try:
        for remaining in range(room.time_per_question - 1, -1, -1):
            await asyncio.sleep(1)
            room.last_activity = time.time()

            if room_manager.get(room.room_code) is not room or room.current_question_index != idx:
                return

            await sio.emit("timer_tick", {
                "question_number": idx + 1,
                "remaining": remaining
            }, room=room.room_code)

            connected = [p for p in room.players.values() if p.is_connected]
            if connected and all((idx + 1) in p.answers for p in connected):
                return

        await show_results(room)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Timer error: {e}")

@sio.on("submit_answer")
async def on_answer(sid, data):
    try:
        if not isinstance(data, dict):
            await sio.emit("error", {"message": "Invalid data"}, to=sid)
            return

        room = room_manager.get(data.get("room_code", ""))
        if not room or room.status != GameStatus.PLAYING:
            await sio.emit("error", {"message": "Game not in progress"}, to=sid)
            return

        player = room.players.get(sid)
        if not player:
            await sio.emit("error", {"message": "Player not found"}, to=sid)
            return

        q_num_raw = data.get("question_number")
        try:
            q_num = int(q_num_raw)
        except (ValueError, TypeError):
            await sio.emit("error", {"message": "Invalid question number"}, to=sid)
            return

        if q_num != room.current_question_index + 1:
            await sio.emit("error", {"message": "Invalid question"}, to=sid)
            return

        q = room.questions[room.current_question_index]
        result = ScoringEngine.calculate(
            data.get("selected_option"), q["correct_option_id"],
            data.get("time_taken_ms", 0), room.time_per_question * 1000,
            player.current_streak
        )

        player.answers[q_num] = result
        player.total_score += result["total_score"]
        player.total_time_ms += data.get("time_taken_ms", 0)
        player.current_streak = result["new_streak"]
        player.max_streak = max(player.max_streak, result["new_streak"])
        if result["is_correct"]:
            player.correct_count += 1
        else:
            player.wrong_count += 1

        await sio.emit("answer_received", {
            "score_gained": result["total_score"],
            "total_score": player.total_score
        }, to=sid)

        connected_players = [p for p in room.players.values() if p.is_connected]
        if all(q_num in p.answers for p in connected_players):
            if room.question_timer_task:
                room.question_timer_task.cancel()
            await show_results(room)
    except Exception as e:
        logger.error(f"Submit answer error: {e}")
        await sio.emit("error", {"message": "Failed to submit answer"}, to=sid)

async def show_results(room: GameRoom):
    room.status = GameStatus.ANSWER_PHASE
    room.last_activity = time.time()
    q = room.questions[room.current_question_index]

    if room.question_timer_task and not room.question_timer_task.done():
        room.question_timer_task.cancel()

    for p in room.players.values():
        if room.current_question_index + 1 not in p.answers:
            p.unanswered_count += 1

    await sio.emit("question_results", {
        "correct_option": q["correct_option_id"],
        "explanation": q["explanation"],
        "player_results": [
            {
                "name": p.name,
                "is_correct": p.answers.get(room.current_question_index + 1, {}).get("is_correct", False),
                "score_gained": p.answers.get(room.current_question_index + 1, {}).get("total_score", 0)
            }
            for p in room.players.values()
        ]
    }, room=room.room_code)

    await asyncio.sleep(2)

    rankings = ScoringEngine.rankings(room.players)
    await sio.emit("leaderboard", {
        "rankings": rankings,
        "next_question_in": 3
    }, room=room.room_code)

    room.status = GameStatus.LEADERBOARD
    await asyncio.sleep(3)

    if room.current_question_index + 1 < len(room.questions):
        await start_question(room, room.current_question_index + 1)
    else:
        await end_game(room)



@sio.on("restart_room")
async def on_restart(sid, data):
    room = room_manager.get_by_sid(sid)
    if not room or not room.players.get(sid, Player("", "", "")).is_host:
        return

    # 1. Reset Room and Player States
    room.status = GameStatus.WAITING
    room.current_question_index = -1
    for p in room.players.values():
        p.total_score = 0
        p.correct_count = 0
        p.wrong_count = 0
        p.unanswered_count = 0
        p.current_streak = 0
        p.max_streak = 0
        p.total_time_ms = 0
        p.answers = {}

    room.questions = []

    # 2. Tell all clients to jump back to the Lobby UI immediately
    await sio.emit("room_restarted", {"room_code": room.room_code}, room=room.room_code)

    # 3. Generate the new exam safely
    try:
        service = QuestionService()
        room.questions = await service.generate(room.subject, room.difficulty, room.chapters)

        for idx, q in enumerate(room.questions):
            q["question_number"] = idx + 1

        # 4. Unlock the Start button
        await sio.emit("questions_ready", {"ready": True}, room=room.room_code)
    except Exception as e:
        logger.error(f"Restart generation failed: {e}")
        await sio.emit("generation_failed", {"message": "AI overloaded, please try again."}, room=room.room_code)

async def end_game(room: GameRoom):
    room.status = GameStatus.FINISHED
    room.last_activity = time.time()
    rankings = ScoringEngine.rankings(room.players)
    room.final_rankings = rankings


    await sio.emit("game_over", {"final_rankings": rankings}, room=room.room_code)

# 12. FASTAPI APP

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache_manager.connect()
    await rate_limiter.connect()
    room_manager.start_cleanup()
    logger.info("Battle Arena API started")
    yield
    global _shared_redis
    if _shared_redis:
        await _shared_redis.close()
        _shared_redis = None
    logger.info("Battle Arena API stopped")

app = FastAPI(
    title="Shik-Shak Arena API",
    version="3.1.0",
    lifespan=lifespan,
    docs_url="/docs" if Config.ENV == "development" else None,
    redoc_url="/redoc" if Config.ENV == "development" else None,
)


app.mount("/socket.io", socket_app)


@app.middleware("http")
async def cors_for_api_only(request, call_next):
    if request.url.path.startswith("/socket.io"):
        return await call_next(request)

    if request.method == "OPTIONS":
        response = Response()
    else:
        response = await call_next(request)

    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3001"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "shik-shak-arena",
        "model": Config.GROQ_MODEL,
        "version": "3.1.0",
        "env": Config.ENV,
    }

@app.post("/api/v1/auth/token")
async def create_token_endpoint(user_id: str = "demo_user", email: str = "demo@example.com"):
    return {
        "access_token": AuthManager.create_token(user_id, email),
        "token_type": "bearer",
        "expires_in": Config.JWT_EXPIRE_MINUTES * 60
    }

@app.post("/api/v1/exams/generate", response_model=ExamResponse)
async def generate_exam(request: ExamGenerationRequest, user: Dict = Depends(rate_limit_dependency)):
    chapters_dict = [ch.model_dump() for ch in request.chapter_mix]
    cached = await cache_manager.get(request.subject.value, request.difficulty.value, chapters_dict)

    if cached:
        questions = [Question(id=i+1, chapter_id=request.chapter_mix[i].id, chapter_name=request.chapter_mix[i].name, **q)
                      for i, q in enumerate(cached["questions"])]
        return ExamResponse(
            subject=request.subject,
            difficulty=request.difficulty,
            questions=questions,
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=True,
            exam_id=cached.get("exam_id")
        )

    service = QuestionService()
    raw_questions = await service.generate(request.subject, request.difficulty, request.chapter_mix)

    await cache_manager.set(request.subject.value, request.difficulty.value, chapters_dict, {"questions": raw_questions})

    questions = [Question(id=i+1, chapter_id=request.chapter_mix[i].id, chapter_name=request.chapter_mix[i].name, **q) for i, q in enumerate(raw_questions)]
    return ExamResponse(
        subject=request.subject,
        difficulty=request.difficulty,
        questions=questions,
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False
    )


# 13. PRACTICE ENDPOINTS (with cleanup)

practice_sessions: Dict[str, Dict] = {}
_practice_cleanup_task: Optional[asyncio.Task] = None

async def cleanup_practice_sessions():
    while True:
        await asyncio.sleep(60)
        now = time.time()

        to_remove = [
            sid for sid, session in practice_sessions.items()
            if now - session.get("created_at", 0) > Config.PRACTICE_SESSION_TTL
        ]
        for sid in to_remove:
            practice_sessions.pop(sid, None)
            logger.info(f"Cleaned up practice session {sid}")

@app.post("/api/v1/practice/start")
async def start_practice(request: ExamGenerationRequest, user: Dict = Depends(AuthManager.get_current_user)):
    service = QuestionService()
    questions = await service.generate(request.subject, request.difficulty, request.chapter_mix)
    session_id = str(uuid.uuid4())

    practice_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user["user_id"],
        "subject": request.subject.value,
        "difficulty": request.difficulty.value,
        "chapter_ids": [ch.id for ch in request.chapter_mix],
        "questions": questions,
        "time_per_question": 60,
        "answers": {},
        "created_at": time.time(),
    }

    global _practice_cleanup_task
    if _practice_cleanup_task is None or _practice_cleanup_task.done():
        _practice_cleanup_task = asyncio.create_task(cleanup_practice_sessions())

    return {
        "session_id": session_id,
        "subject": request.subject.value,
        "difficulty": request.difficulty.value,
        "total_questions": len(questions),
        "time_per_question": 60,
        "questions": [
            {"question_number": i+1, "question_text": q["question_text"], "options": q["options"]}
            for i, q in enumerate(questions)
        ]
    }

@app.post("/api/v1/practice/answer")
async def practice_answer(data: PracticeAnswerRequest, user: Dict = Depends(AuthManager.get_current_user)):
    session = practice_sessions.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    if data.question_number < 1 or data.question_number > len(session["questions"]):
        raise HTTPException(status_code=400, detail="Invalid question number")

    q = session["questions"][data.question_number - 1]
    selected = data.selected_option
    is_correct = selected == q["correct_option_id"] if selected else False
    score = 4 if is_correct else (-1 if selected else 0)

    session["answers"][data.question_number] = {
        "selected": selected,
        "is_correct": is_correct,
        "score": score
    }
    session["last_activity"] = time.time()

    return {
        "is_correct": is_correct,
        "correct_option": q["correct_option_id"],
        "explanation": q["explanation"],
        "score": score
    }

@app.post("/api/v1/practice/finish")
async def finish_practice(data: Dict = Body(...), user: Dict = Depends(AuthManager.get_current_user)):
    session_id = data.get("session_id")
    session = practice_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    answers = session.get("answers", {})
    total = sum(a["score"] for a in answers.values())
    correct = sum(1 for a in answers.values() if a["is_correct"])
    wrong = sum(1 for a in answers.values() if not a["is_correct"] and a["selected"])
    unanswered = len(session["questions"]) - len(answers)
    accuracy = round(correct / len(session["questions"]) * 100, 1) if session["questions"] else 0



    practice_sessions.pop(session_id, None)

    return {
        "session_id": session_id,
        "total_score": total,
        "correct": correct,
        "wrong": wrong,
        "unanswered": unanswered,
        "accuracy": accuracy,
    }

@app.get("/api/v1/practice/{session_id}/results")
async def practice_results(session_id: str, user: Dict = Depends(AuthManager.get_current_user)):
    session = practice_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your session")

    answers = session["answers"]
    total = sum(a["score"] for a in answers.values())
    correct = sum(1 for a in answers.values() if a["is_correct"])
    wrong = len(answers) - correct
    unanswered = len(session["questions"]) - len(answers)

    return {
        "session_id": session_id,
        "total_score": total,
        "correct_count": correct,
        "wrong_count": wrong,
        "unanswered_count": unanswered,
        "accuracy": round(correct / len(session["questions"]) * 100, 1) if session["questions"] else 0,
        "completed": len(answers) >= len(session["questions"]),
    }

@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)