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



