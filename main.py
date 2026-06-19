import os
import json
from enum import Enum
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from sympy import symbols, simplify
from sympy.parsing.latex import parse_latex

load_dotenv()



# 1. DOMAIN DATA MODELS & ENUMS


class Subject(str, Enum):
    MATH = "MATH"
    PHYSICS = "PHYSICS"
    CHEMISTRY = "CHEMISTRY"
    BIOLOGY = "BIOLOGY"


class Difficulty(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


class ChapterMixItem(BaseModel):
    id: str
    name: str


class ExamGenerationRequest(BaseModel):
    subject: Subject
    difficulty: Difficulty
    chapter_mix: List[ChapterMixItem] = Field(..., max_items=5, min_items=5)



