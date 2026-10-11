import os
import re
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
client = (
    OpenAI(api_key=token, base_url="https://aipipe.org/openrouter/v1")
    if token else None
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
        return {"success": True, "output": stdout_buffer.getvalue()}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_error_lines(error_traceback: str) -> List[int]:
    # Use frames from the submitted code, not frames from this API.
    matches = re.findall(r'File "<string>", line (\d+)', error_traceback)
    return [int(matches[-1])] if matches else []


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    if client is None:
        return []

    prompt = f"""
Identify the line where the exception occurs in the submitted Python code.
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
                    "Analyze Python tracebacks accurately. Return valid JSON only, "
                    'using the schema {"error_lines": [1]}.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )
    line_count = len(code.splitlines())
    return sorted({n for n in result.error_lines if 1 <= n <= line_count})


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    error_traceback = execution["output"]
    error_lines = get_error_lines(error_traceback)

    if not error_lines:
        try:
            error_lines = analyze_error_with_ai(request.code, error_traceback)
        except Exception:
            error_lines = []

    return {"error": error_lines, "result": error_traceback}


@app.get("/")
def home():
    return {"message": "AI Code Interpreter is running"}