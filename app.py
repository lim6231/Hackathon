import os
import json
from flask import Flask, request, render_template_string
import openai

# ----------- SAFETY CHECK -----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

openai.api_key = api_key

# ----------- AGENT FUNCTION -----------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def generate_test_plan(user_text, memory):
    memory.append({"role": "user", "content": user_text})
    # Build messages in old v0.28 format
    conversation = ""
    for msg in memory:
        role = msg["role"]
        conversation += f"{role}: {msg['content']}\n"

    prompt = f"""
You are an AI Test Coverage Optimizer.
Input user stories, requirements, log data, or past defects.
Generate:
- risk scores
- most impactful test cases
- missing coverage areas
- prioritized test plan

User input and conversation history:
{conversation}

Output concisely and clearly.
"""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=800
    )

    reply = response.choices[0].text.strip()
    memory.append({"role": "assistant", "content": reply})
    save_memory(memory)
    return reply

# ----------- FLASK APP -----------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>
<form method="post">
<input name="user_input" size="80"/>
<input type="submit" value="Send"/>
</form>
<div style="margin-top:20px;">
{% for entry in history %}
<p><b>{{ entry.role }}:</b> {{ entry.content }}</p>
{% endfor %}
</div>
</body>
</html>
"""

chat_history = load_memory()

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            chat_history.append({"role": "user", "content": user_input})
            reply = generate_test_plan(user_input, chat_history)
            chat_history.append({"role": "assistant", "content": reply})
    return render_template_string(HTML_PAGE, history=chat_history)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
