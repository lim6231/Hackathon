import os
import json
from flask import Flask, request, render_template_string
import openai

# ----------- SAFETY CHECK -----------
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI Test Coverage Optimizer."},
        {"role": "user", "content": "Some user stories here"}
    ]
)
# ----------- FLASK APP -----------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>
<form method="post">
<textarea name="user_input" rows="6" cols="80" placeholder="Paste user stories, requirements, log data, or past defects here..."></textarea><br>
<input type="submit" value="Generate Test Plan"/>
</form>
<div style="margin-top:20px;">
{% for entry in history %}
<p><b>{{ entry.role }}:</b> {{ entry.content }}</p>
{% endfor %}
</div>
</body>
</html>
"""

chat_history = []

# ----------- CORE AI FUNCTION -----------
def generate_test_plan(user_input: str) -> str:
    """
    Calls OpenAI to generate prioritized test plan with risk scores,
    impactful test cases, and missing coverage areas.
    """
    system_prompt = (
        "You are an AI Test Coverage Optimizer for enterprise QA teams. "
        "Given user stories, requirements, log data, or past defects, "
        "output a structured prioritized test plan including:\n"
        "- Risk scores\n"
        "- Most impactful test cases\n"
        "- Missing coverage areas\n"
        "Format the output clearly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

# ----------- ROUTES -----------
@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            chat_history.append({"role": "You", "content": user_input})
            try:
                reply = generate_test_plan(user_input)
            except Exception as e:
                reply = f"[ERROR] {e}"
            chat_history.append({"role": "[AI Test Coverage Optimizer]", "content": reply})

    return render_template_string(HTML_PAGE, history=chat_history)

# ----------- MAIN -----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
