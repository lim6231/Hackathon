# optimizer.py
import json
import re
import time
from openai import OpenAI
import openai as openai_lib  # for exceptions

def safe_openai_call(func, *args, retries=4, initial_delay=2, **kwargs):
    """
    Generic retry wrapper for OpenAI API calls on rate limits or transient errors.
    """
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except openai_lib.error.RateLimitError as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except openai_lib.error.APIConnectionError as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2

def _clean_json_from_text(text: str):
    """
    Try to extract the first JSON object/array from model text.
    """
    # simple heuristic: find first { and last } matching
    text = text.strip()
    # If text is pure JSON, return it
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text
    # Try to extract using regex for a JSON object
    m = re.search(r"(\{(?:.|\n)*\})", text)
    if m:
        return m.group(1)
    # fallback: try to find array
    m = re.search(r"(\[(?:.|\n)*\])", text)
    if m:
        return m.group(1)
    return None

def analyze_artifacts(client: OpenAI, artifacts: dict, model: str = "gpt-4o-mini") -> dict:
    """
    Given artifacts (user_stories and context), call OpenAI to generate:
      - risk_scores: list of {area, score, reason}
      - missing_coverage: list of descriptions
      - most_impactful_tests: list of tests (title, description, reason)
      - prioritized_plan: ordered list of test cases with priority and effort estimate

    Returns a structured dict.
    """
    system_prompt = (
        "You are an expert test strategist for enterprise software. "
        "Given user stories, requirements, logs, and defect history, produce a structured JSON report with: "
        "1) risk_scores: an array of objects {area, score(0-100), rationale} "
        "2) missing_coverage: array of short descriptions of uncovered areas "
        "3) most_impactful_tests: array of objects {id, title, description, impact} "
        "4) prioritized_plan: array of objects {id, priority(P1/P2/P3), estimated_hours, reason} "
        "Output ONLY valid JSON (no extra commentary) using these exact keys."
    )

    user_text = artifacts.get("user_stories", "")
    context_text = artifacts.get("context", "")

    prompt = (
        f"{system_prompt}\n\n"
        "INPUT DATA:\n"
        f"User stories / requirements:\n{user_text}\n\n"
        "Context (logs, past defects):\n"
        f"{context_text}\n\n"
        "If some sections are empty, still produce reasonable defaults and assumptions. Keep each 'most_impactful_tests' item concise (title + 1-2 sentence description)."
    )

    # message payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1200,
    }

    # call with safety wrapper
    resp = safe_openai_call(client.chat.completions.create, **kwargs)
    # resp.choices[0].message.content is the assistant output (object-style)
    raw = getattr(resp.choices[0].message, "content", None)
    if raw is None:
        # try alternative access (older clients)
        raw = resp.choices[0]["message"]["content"]

    # attempt to extract JSON
    json_text = _clean_json_from_text(raw)
    if not json_text:
        # if model didn't return direct JSON, try a secondary prompt to force JSON
        follow_prompt = (
            "The previous output was not strict JSON. Please reformat your answer to be valid JSON with only the keys: "
            "risk_scores, missing_coverage, most_impactful_tests, prioritized_plan. Use the original analysis to populate them."
        )
        follow = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": follow_prompt + "\n\nOriginal output:\n" + raw}
        ]
        resp2 = safe_openai_call(client.chat.completions.create, **{"model": model, "messages": follow, "temperature": 0.0, "max_tokens": 800})
        raw2 = getattr(resp2.choices[0].message, "content", None) or resp2.choices[0]["message"]["content"]
        json_text = _clean_json_from_text(raw2)
        if not json_text:
            raise ValueError("Could not parse JSON from model output.")

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        # last-ditch: attempt to fix common issues (trailing commas)
        fixed = re.sub(r",\s*}", "}", json_text)
        fixed = re.sub(r",\s*]", "]", fixed)
        parsed = json.loads(fixed)

    # ensure keys exist
    parsed.setdefault("risk_scores", [])
    parsed.setdefault("missing_coverage", [])
    parsed.setdefault("most_impactful_tests", [])
    parsed.setdefault("prioritized_plan", [])

    return parsed
