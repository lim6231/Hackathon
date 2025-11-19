import os
import json
import requests
from flask import Flask, request, render_template_string
import openai

# ----------- SAFETY CHECK -----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

# Use OpenAI v0.28
openai.api_key = api_key

# ----------- YOUR ORIGINAL FUNCTIONS -----------
def http_get(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        return r.text[:3000]
    except Exception as e:
        return f"HTTP_GET_ERROR: {e}"

def echo(text: str) -> str:
    return f"ECHO_RESULT: {text}"

# ----------- AI TEST COVERAGE OPTIMIZER -----------
def generate_test_plan(user_input: str) -> str:
    prompt = f"""
You are an AI Test Coverage Optimizer.
Input: {user_input}
Output a structured prioritized test plan with:
1. Risk scores
2. Most impactful test cases
3. Missing coverage areas
Format it clearly for display.
"""
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=800,
        temperature=0.3
    )
    return response.choices[0].text.strip()

# ----------- FLASK APP -----------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>
<form method="post">
<textarea name="user_input" rows="6" cols="80" placeholder="Paste requirements, defects, or logs here..."></textarea><br>
<input type="submit" value="Generate Test Plan"/>
</form>
<div style="margin-top:20px; white-space: pre-wrap;">
{% if output %}
<h3>Generated Test Plan:</h3>
<p>{{ output }}</p>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    output = None
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            output = generate_test_plan(user_input)
    return render_template_string(HTML_PAGE, output=output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
