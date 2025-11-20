import os
import json
import requests
from flask import Flask, request, render_template_string, session
from openai import OpenAI
from uuid import uuid4

# ----------- OPENAI CLIENT INITIALIZATION -----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------- TOOLS -----------
def http_get(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        return r.text[:3000]
    except Exception as e:
        return f"HTTP_GET_ERROR: {e}"

def echo(text: str) -> str:
    return f"ECHO_RESULT: {text}"

# ----------- AGENT CLASS -----------
class Agent:
    def __init__(self, name: str, system_prompt: str, tools=None, model="gpt-4o-mini", memory_file=None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.model = model
        self.memory_file = memory_file or f"{self.name}_memory.json"
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def _openai_call(self, messages, functions=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 800
        }
        if functions:
            kwargs["functions"] = functions
            kwargs["function_call"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message

    def handle(self, user_text: str, session_memory=None) -> str:
        session_memory = session_memory if session_memory is not None else []

        session_memory.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": self.system_prompt}] + session_memory

        msg = self._openai_call(messages)

        if not hasattr(msg, "function_call") or msg.function_call is None:
            reply = msg.content.strip()
        else:
            fn = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")

            if fn not in self.tools:
                reply = f"[ERROR] Unknown tool requested: {fn}"
            else:
                try:
                    result = self.tools[fn](**args)
                except Exception as e:
                    result = f"[TOOL_ERROR] {e}"

                follow_messages = (
                    [{"role": "system", "content": self.system_prompt}]
                    + session_memory
                    + [
                        {"role": "assistant", "content": f"[Function {fn} executed]"},
                        {"role": "function", "name": fn, "content": result}
                    ]
                )

                final = self._openai_call(follow_messages)
                reply = final.content.strip()

        session_memory.append({"role": "assistant", "content": reply})
        self.memory.extend(session_memory)
        self.save_memory()
        return reply

# ----------- FLASK APP -----------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or str(uuid4())

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>Hello</h2>
<form method="post">
<textarea name="user_input" rows="5" cols="80" placeholder="Enter multiple user stories separated by line breaks"></textarea><br>
<input type="submit" value="Send"/>
</form>
<div style="margin-top:20px;">
{% if table %}
<h3>Prioritized Test Plan</h3>
{{ table|safe }}
{% endif %}
{% for entry in history %}
<p><b>{{ entry.role }}:</b> {{ entry.content }}</p>
{% endfor %}
</div>
</body>
</html>
"""

agent = Agent(
    name="test_optimizer",
    system_prompt=(
        "You are the 'AI Test Coverage Optimizer'. "
        "Your ONLY job is to: "
        "- take any input (user stories, requirements, logs, or questions) "
        "- extract implied features, risks, and coverage gaps "
        "- generate detailed test plans with the following fields per test case: "
        "  * risk (1–5), "
        "  * functional_area, "
        "  * test_case_steps (numbered steps specifically for SCCM CMG deployment, based on official Microsoft documentation or standard ConfigMgr procedures, like '1. Create CMG...', '2. Deploy application...', etc.) "
        "  * expected_result, "
        "  * missing_coverage, "
        "  * rationale (why this test is important) "
        "- ALWAYS output valid JSON ONLY with structure: "
        "{ 'plan': [ { 'risk': 5, 'functional_area': '...', "
        "'test_case_steps': ['step1', 'step2'], 'expected_result': '...', "
        "'missing_coverage': '...', 'rationale': '...' } ] } "
        "NEVER give general explanations or text outside JSON."
    ),
    tools={"http_get": http_get, "echo": echo}
)


@app.route("/", methods=["GET", "POST"])
def chat():
    if "session_memory" not in session:
        session["session_memory"] = []
    
    table_html = None
    chat_history = session.get("chat_history", [])

    if request.method == "POST":
        user_input = request.form.get("user_input", "")

        if user_input:
            sccm_reference = """
            You need to generate a test plan for SCCM CMG deployment.
            Reference: https://learn.microsoft.com/en-us/intune/configmgr"""
            combined_input = sccm_reference + "\nUser query: " + user_input
            chat_history.append({"role": "You", "content": user_input})
            reply = agent.handle(combined_input, session_memory=session["session_memory"])
    
            try:
                data = json.loads(reply)
                plan = data.get("plan", [])
                if plan:    
                    rows = ""
                    for p in plan:
                        steps = "<br>".join(p.get("test_case_steps", []))
                        rows += (
                            "<tr>"
                            f"<td>{p['risk']}</td>"
                            f"<td>{p.get('functional_area', '')}</td>"
                            f"<td>{steps}</td>"
                            f"<td>{p.get('expected_result', '')}</td>"
                            f"<td>{p.get('missing_coverage', '')}</td>"
                            f"<td>{p.get('rationale', '')}</td>"
                            "</tr>"
                        )
                table_html = (
                    "<table border='1'>"
                    "<tr><th>Risk Score</th><th>Functional Area</th><th>Test Steps</th>"
                    "<th>Expected Result</th><th>Missing Coverage</th><th>Rationale</th></tr>"
                    f"{rows}</table>"
                )
            except:
                table_html = reply


    session["session_memory"].append({"role": "assistant", "content": table_html})
    chat_history.append({"role": "assistant", "content": table_html})
    return render_template_string(HTML_PAGE, history=chat_history, table=table_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
