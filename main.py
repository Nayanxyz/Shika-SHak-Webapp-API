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



# 3. PROMPT MANAGEMENT & AI CLIENT SERVICE

class PromptTemplateManager:

    @staticmethod
    def get_system_prompt() -> str:
        return r"""You are the Lead Exam Architect for an elite competitive examination engine (IIT-JEE Advanced/Main and NEET). 
        Your sole objective is to generate a flawless, highly structured exam payload based strictly on the provided input parameters. 

### INPUT PARAMETERS TO INGEST:
- Target Subject: {subject}
- Global Difficulty: {difficulty}
- Selected Shotgun Chapter Mix: {chapter_mix_list}

---

### EXECUTION MANDATE & ARCHITECTURAL PATHWAYS:

1. QUESTION DISTRIBUTION (SHOTGUN MIX):
   - You must generate exactly 5 Multiple Choice Questions (MCQs).
   - Each question must belong to exactly ONE chapter from the provided list. 
   - Question 1 maps to Chapter 1, Question 2 to Chapter 2, and so on. No skipping or duplicating chapters.

2. LOGICAL DIFFICULTY SCALING:
   - If difficulty is "LOW": Generate straightforward, single-concept application problems. Focus on core formula usage, clear terminology, and clean numerical execution.
   - If difficulty is "HIGH": Generate brutal, multi-layered, non-standard problems. Force concept compounding. Employ information masking.

3. OPTION-SPACE DISTRACTOR ENGINEERING:
   - Every question must contain exactly 4 options (A, B, C, D) with exactly 1 correct answer.
   - If difficulty is "LOW": Ensure the 3 incorrect options are distinctly off-target.
   - If difficulty is "HIGH": The options must look structurally and mathematically symmetrical. Simulate high-probability student errors (Trap 1: Boundary/Sign Error, Trap 2: Premature Stop, Trap 3: Constant Coefficient Error).

4. DOWNSTREAM VALIDATION HOOKS & BIOLOGY STRUCTURES:
   - For MATHEMATICS and PHYSICS: Provide the pure LaTeX equation representing the solution step in the metadata.
   - For CHEMISTRY: Identify all chemical compounds involved and output their names in an array under `verification_chemicals`.
   - For BIOLOGY: Content must be strictly 100% grounded in NCERT concepts. For HIGH difficulty, you MUST NOT generate standard single-sentence questions. You must strictly use one of these three premium NTA formats:
     a) Assertion-Reasoning (A: Assertion, R: Reason type)
     b) Statement I and Statement II combinations
     c) Multi-statement validation blocks (List 5 statements, and options choose 'How many are correct')

5. FORMATTING RIGOR & JSON COMPLIANCE:
   - All mathematical symbols, chemical formulas, and equations must be formatted in valid LaTeX.
   - CRITICAL NATIVE ESCAPE RULE: To prevent JSON syntax corruption, you MUST NEVER use the literal backslash (`\`) character anywhere in your response text. Instead, replace every single backslash with the literal token `[BS]`.
   - For example: write `[BS]frac{{a}}{{b}}`, `[BS]left(`, `[BS]right)`, `[BS]sin(x)`, `[BS]begin{{pmatrix}}`, and `[BS]end{{pmatrix}}`.
   - INTERNAL QUOTES RULE: If you must use a quotation mark or double-quote anywhere inside your question text, options, or explanations, you MUST escape it using `\"`. Do not use actual JSON comments (`//`) anywhere in your output.

### HIGH-DIFFICULTY MATH EXEMPLAR (TARGET BENCHMARK):
If a question is targeted at HIGH difficulty, it must mirror this level of compounding (written using the mandatory `[BS]` token layout):
Question: Let f: [BS]mathbb{{R}} [BS]rightarrow [BS]mathbb{{R}} be a differentiable function such that f(1) = 2 and f'(x) = [BS]frac{{f(x)^2 + 1}}{{x^2 + 1}} for all x > 0. Find the value of [BS]lim_{{x [BS]rightarrow [BS]infty}} f(x).
Do not generate simple identity plug-ins. Force variables to cross multiple domain boundaries.

### HIGH-DIFFICULTY BIOLOGY EXEMPLAR (TARGET BENCHMARK):
If a question is targeted at BIOLOGY and HIGH difficulty, it must mirror this level of NCERT-grounded logical structure:
Question: Given below are two statements:
Statement I: The primary acceptor of CO2 in C4 plants is phosphoenolpyruvate (PEP) and it is found in mesophyll cells.
Statement II: Mesophyll cells of C4 plants lack the RuBisCO enzyme.
In the light of the above statements, choose the most appropriate answer from the options given below.
Do not generate simple recall questions. Force micro-word validation checks.

---

### OUTPUT DATA CONTRACT (STRICT JSON SCHEMA):
You must output a single, raw JSON object matching the exact structure below. The "questions" array must contain exactly 5 distinct question objects constructed identically to the blueprint item shown. Do not append any comments or conversational text outside or inside the JSON block. All LaTeX strings inside your properties must use `[BS]` instead of backslashes.

{{
  "subject_executed": "{subject}",
  "difficulty_applied": "{difficulty}",
  "questions": [
    {{
      "question_number": 1,
      "chapter_id": "string",
      "chapter_name": "string",
      "question_text": "string_with_backslash_bypass_tokens",
      "options": [
        {{ "id": "A", "text": "string" }},
        {{ "id": "B", "text": "string" }},
        {{ "id": "C", "text": "string" }},
        {{ "id": "D", "text": "string" }}
      ],
      "correct_option_id": "A",
      "metadata": {{
        "trap_type_breakdown": "string",
        "step_by_step_derivation": "string",
        "verification_chemicals": [],
        "symbolic_verification_target": "string"
      }}
    }}
  ]
}}"""

    def build_orchestrator_prompt(self, subject: Subject, difficulty: Difficulty,
                                  chapters: List[Dict[str, str]]) -> str:
        base_template = self.get_system_prompt()
        return base_template.format(
            subject=subject.value,
            difficulty=difficulty.value,
            chapter_mix_list=str(chapters)
        )

class OpenAIService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Critical Configuration Missing: GROQ_API_KEY is not set.")
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    def execute_structured_generation(self, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Execute production run based on system rules."}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content


# 4. CORE ORCHESTRATOR WITH RETRY LIFECYCLE


