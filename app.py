import os
import json
import requests
from flask import Flask, request, render_template_string, session
from flask_session import Session
from openai import OpenAI
from uuid import uuid4

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def http_get(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        return r.text[:3000]
    except Exception as e:
        return f"HTTP_GET_ERROR: {e}"

def echo(text: str) -> str:
    return f"ECHO_RESULT: {text}"

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

# ---------------- Flask app ----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or str(uuid4())

# server-side sessions to avoid cookie size limit
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session_files"
app.config["SESSION_PERMANENT"] = False
Session(app)

# folder for uploaded files
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>

<form method="post" enctype="multipart/form-data">
<textarea name="user_input" rows="5" cols="80" placeholder="Enter multiple user stories separated by line breaks"></textarea><br>
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
        "Your ONLY job is to: "
        "- take any input (user stories, requirements, logs, or questions) "
        "- extract implied features, risks, and coverage gaps "
        "- generate detailed test plans with fields: risk (1–5), functional_area, test_case_steps, expected_result, missing_coverage, rationale "
        "- ALWAYS output valid JSON ONLY with structure: "
        "{ 'plan': [ { 'risk': 5, 'functional_area': '...', 'test_case_steps': ['step1'], 'expected_result': '...', 'missing_coverage': '...', 'rationale': '...' } ] } "
        "NEVER give general explanations or text outside JSON."
    ),
    tools={"http_get": http_get, "echo": echo}
)

@app.route("/", methods=["GET", "POST"])
def chat():
    if "session_memory" not in session:
        session["session_memory"] = []

    chat_history_file = "chat_history.json"
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

    table_html = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "") or ""
        uploaded_file = request.files.get("file")
        url_input = request.form.get("url", "").strip()
        save_k = request.form.get("save_knowledge") == "on"

        # knowledge context
        knowledge_items = load_knowledge()
        knowledge_block = ""
        if knowledge_items:
            knowledge_block = "\n\n--- STORED KNOWLEDGE ---\n" + "\n\n".join(knowledge_items)

        transient_sources = []

        # handle URL
        if url_input:
            fetched = http_get(url_input)
            transient_sources.append(f"[URL CONTENT FROM {url_input}]\n{fetched}")
            if save_k:
                add_knowledge(f"[URL {url_input}]\n{fetched}")

        # handle uploaded file (saved to disk)
        if uploaded_file:
            filename = uploaded_file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(file_path)
            transient_sources.append(f"[UPLOADED FILE: {filename}] saved at {file_path}")
            if save_k:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_text = f.read()
                    add_knowledge(f"[FILE {filename}]\n{file_text}")
                except:
                    add_knowledge(f"[FILE {filename}]\n[UNREADABLE]")

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

        # try to render JSON → table
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
                chat_history.append({"role": "assistant", "content": table_html})
            else:
                chat_history.append({"role": "assistant", "content": reply})
        except Exception:
            chat_history.append({"role": "assistant", "content": reply})

        # persist chat history on disk
        with open(chat_history_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)

    return render_template_string(HTML_PAGE, history=chat_history, table=table_html)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("./flask_session_files", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
