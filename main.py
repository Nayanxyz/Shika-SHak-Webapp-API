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

            for i in range(len(parsed_exprs)):
                for j in range(i + 1, len(parsed_exprs)):
                    id1, expr1 = parsed_exprs[i]
                    id2, expr2 = parsed_exprs[j]

                    try:
                        # If algebraic subtraction simplifies cleanly to 0, they are identical
                        if simplify(expr1 - expr2) == 0:
                            print(
                                f"SymPy detected mathematically identical choices: {id1} and {id2}")
                            return False
                    except Exception:
                        continue

        return True


class ChemistryValidator:

    def validate(self, raw_questions: List[Dict[str, Any]]) -> bool:
        print("Bypassing PubChem network verification; system active.")
        return True


class BiologyValidator:

    def validate(self, raw_questions: List[Dict[str, Any]]) -> bool:
        print("Bypassing local ChromaDB vector comparison; system active.")
        return True


class ValidationRouter:

    def __init__(self):
        self.math_physics = MathPhysicsValidator()
        self.chemistry = ChemistryValidator()
        self.biology = BiologyValidator()

    def route_and_verify(self, subject: Subject, raw_questions: List[Dict[str, Any]]) -> bool:
        if subject in [Subject.MATH, Subject.PHYSICS]:
            return self.math_physics.validate(raw_questions)
        elif subject == Subject.CHEMISTRY:
            return self.chemistry.validate(raw_questions)
        elif subject == Subject.BIOLOGY:
            return self.biology.validate(raw_questions)
        return False



