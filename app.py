# app.py
import os
import json
import re
import time
from flask import Flask, request, render_template, send_file, jsonify
from openai import OpenAI
from optimizer import analyze_artifacts, safe_openai_call
from io import BytesIO

app = Flask(__name__, template_folder="templates")

# OpenAI client (reads OPENAI_API_KEY from environment)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in environment before running.")
client = OpenAI(api_key=api_key)

# Configuration: default model (can override via env)
DEFAULT_MODEL = os.getenv("OT_MODEL", "gpt-4o-mini")

# Home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

# Analyze endpoint (form submit)
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts:
      - pasted text (user_stories)
      - optional 'context' field (logs, defects)
    Returns rendered HTML with JSON result and a download link.
    """
    user_text = request.form.get("user_text", "").strip()
    context_text = request.form.get("context_text", "").strip()
    model = request.form.get("model", DEFAULT_MODEL)

    if not user_text and not context_text:
        return render_template("index.html", error="Provide user stories, requirements, or defects to analyze.", result=None)

    # Compose artifacts
    artifacts = {
        "user_stories": user_text,
        "context": context_text
    }

    try:
        result = analyze_artifacts(client=client, artifacts=artifacts, model=model)
    except Exception as e:
        # surface error to user
        return render_template("index.html", error=str(e), result=None)

    # Save JSON into session-like memory (in-memory) and also provide downloadable content
    json_bytes = json.dumps(result, indent=2).encode("utf-8")
    download_token = str(int(time.time() * 1000))
    # store temporarily in /tmp with token name
    tmp_path = f"/tmp/coverage_report_{download_token}.json"
    with open(tmp_path, "wb") as f:
        f.write(json_bytes)

    return render_template("index.html", result=result, download_token=download_token)

# Download route: /download/<token>
@app.route("/download/<token>", methods=["GET"])
def download(token):
    tmp_path = f"/tmp/coverage_report_{token}.json"
    if not os.path.exists(tmp_path):
        return "No report found or it expired.", 404
    return send_file(tmp_path, as_attachment=True, download_name=f"coverage_report_{token}.json")

# Minimal API for automation: POST /api/analyze
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(force=True)
    artifacts = {
        "user_stories": payload.get("user_stories", ""),
        "context": payload.get("context", "")
    }
    model = payload.get("model", DEFAULT_MODEL)
    try:
        result = analyze_artifacts(client=client, artifacts=artifacts, model=model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
