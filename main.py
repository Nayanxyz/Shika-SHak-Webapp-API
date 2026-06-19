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



# 2. VALIDATION TRACKS (The Guardrail Engines)

class MathPhysicsValidator:

    def __init__(self):
        self.symbols_map = symbols('x y z a b c')

    def validate(self, raw_questions: List[Dict[str, Any]]) -> bool:
        for q in raw_questions:
            options = q.get("options", [])


            option_texts = [opt["text"].replace("$", "").strip() for opt in options]
            if len(set(option_texts)) != len(options):
                print("[Validation Failed] Exact duplicate text string detected in option space.")
                return False


            parsed_exprs = []
            for opt in options:
                try:
                    clean_latex = opt["text"].replace("$", "").strip()


                    if any(token in clean_latex for token in ["=", "or", "and", "lim", "matrix", "begin"]):
                        continue

                    expr = parse_latex(clean_latex)
                    parsed_exprs.append((opt["id"], expr))
                except Exception:
                    print(f"SymPy skipped parsing option {opt['id']} (Advanced Notation).")
                    continue

