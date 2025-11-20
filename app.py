import os
import json
import requests
from flask import Flask, request, render_template_string, session
from flask_session import Session
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

        reply = msg.content.strip()
        session_memory.append({"role": "assistant", "content": reply})
        self.memory.extend(session_memory)
        self.save_memory()
        return reply

# --------- Knowledge store helpers ----------
KNOWLEDGE_FILE = "knowledge_base.json"

def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("items", [])
        except:
            return []
    return []

def save_knowledge(items):
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)

def add_knowledge(text):
    items = load_knowledge()
    items.append(text)
    save_knowledge(items)

# ----------- FLASK APP -----------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or str(uuid4())

# Use server-side session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
Session(app)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>

<form method="post" enctype="multipart/form-data">
<textarea name="user_input" rows="5" cols="80" placeholder="Enter user stories or queries"></textarea><br>
<input type="file" name="file"><br>
<input type="text" name="url" placeholder="Enter URL (OneNote, docs, etc.)"><br>
<label><input type="checkbox" name="save_knowledge"> Save uploaded/URL content to knowledge base</label><br>
<input type="submit" value="Send"/>
</form>

<div style="margin-top:20px;">
{% if table %}
<h3>Prioritized Test Plan</h3>
{{ table|safe }}
{% endif %}

{% for entry in history %}
<p><b>{{ entry.role }}:</b> {{ entry.content|safe }}</p>
{% endfor %}
</div>
</body>
</html>
"""

agent = Agent(
    name="test_optimizer",
    system_prompt=(
        "You are the 'Test Coverage Optimizer'. "
        "Your job is to take user input (requirements, logs, questions) and generate a detailed SCCM CMG test plan. "
        "Output JSON only: { 'plan':[{'risk':int,'functional_area':str,'test_case_steps':[str],"
        "'expected_result':str,'missing_coverage':str,'rationale':str}] }"
    ),
    tools={"http_get": http_get, "echo": echo}
)

@app.route("/", methods=["GET", "POST"])
def chat():
    if "session_memory" not in session:
        session["session_memory"] = []

    chat_history = session.get("chat_history", [])
    table_html = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "") or ""
        uploaded_file = request.files.get("file")
        url_input = request.form.get("url", "").strip()
        save_k = request.form.get("save_knowledge") == "on"

        knowledge_items = load_knowledge()
        knowledge_block = "\n\n--- STORED KNOWLEDGE ---\n" + "\n\n".join(knowledge_items) if knowledge_items else ""

        transient_sources = []

        if url_input:
            fetched = http_get(url_input)
            transient_sources.append(f"[URL {url_input}]\n{fetched}")
            if save_k:
                add_knowledge(f"[URL {url_input}]\n{fetched}")

        if uploaded_file:
            try:
                file_bytes = uploaded_file.read()
                file_text = file_bytes.decode(errors="ignore")
            except Exception:
                file_text = "[UNABLE TO READ UPLOADED FILE]"
            transient_sources.append(f"[FILE {uploaded_file.filename}]\n{file_text}")
            if save_k:
                add_knowledge(f"[FILE {uploaded_file.filename}]\n{file_text}")

        sccm_reference = (
            "You need to generate a test plan for SCCM CMG deployment.\n"
            "Reference: https://learn.microsoft.com/en-us/intune/configmgr"
        )

        combined_parts = []
        if knowledge_block:
            combined_parts.append(knowledge_block)
        if transient_sources:
            combined_parts.append("\n\n--- SUBMISSION SOURCES ---\n" + "\n\n".join(transient_sources))
        combined_parts.append("\n\n--- SCCM REFERENCE ---\n" + sccm_reference)
        combined_parts.append("\n\n--- USER QUERY ---\n" + user_input)
        combined = "\n\n".join(combined_parts)

        chat_history.append({"role": "You", "content": user_input})
        reply = agent.handle(combined, session_memory=session["session_memory"])

        parsed_ok = False
        try:
            data = json.loads(reply)
            plan = data.get("plan", [])
            rows = ""
            for p in plan:
                steps = "<br>".join(p.get("test_case_steps", []))
                rows += (
                    "<tr>"
                    f"<td>{p.get('risk', '')}</td>"
                    f"<td>{p.get('functional_area', '')}</td>"
                    f"<td>{steps}</td>"
                    f"<td>{p.get('expected_result', '')}</td>"
                    f"<td>{p.get('missing_coverage', '')}</td>"
                    f"<td>{p.get('rationale', '')}</td>"
                    "</tr>"
                )
            if rows:
                table_html = (
                    "<table border='1'>"
                    "<tr><th>Risk Score</th><th>Functional Area</th><th>Test Steps</th>"
                    "<th>Expected Result</th><th>Missing Coverage</th><th>Rationale</th></tr>"
                    f"{rows}</table>"
                )
                parsed_ok = True
        except Exception:
            parsed_ok = False

        chat_history.append({"role": "assistant", "content": table_html if parsed_ok else reply})
        session["chat_history"] = chat_history

    return render_template_string(HTML_PAGE, history=chat_history, table=table_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
