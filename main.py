import os
import sys
import traceback
from io import StringIO
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Code Interpreter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

token = os.getenv("AIPIPE_TOKEN")

client = None
if token:
    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1",
    )


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        exec(compile(code, "<string>", "exec"), {})

        return {
            "success": True,
            "output": stdout_buffer.getvalue(),
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(
    code: str,
    error_traceback: str,
) -> List[int]:
    if client is None:
        return []

    prompt = f"""
Analyze the submitted Python code and traceback.
Identify the exact error line numbers in the submitted code.
Count the first code line as line 1.
Return a JSON object with an integer array named error_lines.
Do not invent line numbers.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Python traceback analyst. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    result = ErrorAnalysis.model_validate_json(content)

    line_count = len(code.splitlines())

    return sorted(
        {
            line
            for line in result.error_lines
            if 1 <= line <= line_count
        }
    )


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception:
        error_lines = []

    return {
        "error": error_lines,
        "result": execution["output"],
    }


@app.get("/")
def home():
    return {"message": "AI Code Interpreter is running"}
