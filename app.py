import os
import json
import requests
from flask import Flask, request, render_template_string
from openai import OpenAI

# ----------- SAFETY CHECK -----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

client = OpenAI(api_key=api_key)


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

    # ------------ PATCHED MEMORY-SAFE VERSION ------------
    def handle(self, user_text: str) -> str:

        # Save user message
        self.memory.append({"role": "user", "content": user_text})

        # Build conversation with memory
        messages = [{"role": "system", "content": self.system_prompt}] + self.memory

        msg = self._openai_call(messages)

        # No function call
        if not hasattr(msg, "function_call") or msg.function_call is None:
            reply = msg.content.strip()

        # Function call flow
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

                # Follow-up messages (NOT stored to memory)
                follow_messages = (
                    [{"role": "system", "content": self.system_prompt}] +
                    self.memory +
                    [
                        {"role": "assistant", "content": f"[Function {fn} executed]"},
                        {"role": "function", "name": fn, "content": result}
                    ]
                )

                final = self._openai_call(follow_messages)
                reply = final.content.strip()

        # Save clean assistant response only
        self.memory.append({"role": "assistant", "content": reply})
        self.save_memory()

        return reply


# ----------- ROUTER -----------
class Router:
    def __init__(self):
        self.labels = {
            "infra": "SCCM, SUP, ADR, WSUS, DP, Windows updates, infra topics",
            "coding": "SQL, Java, C#, Selenium, code/debugging",
            "general": "normal IT/misc questions"
        }
        labels_txt = "\n".join([f"- {k}: {v}" for k, v in self.labels.items()])

        self.agent = Agent(
            name="router",
            system_prompt=(
                "You are a router. Choose exactly one label for the user message.\n"
                "Labels:\n" + labels_txt +
                "\nONLY respond with the label name. No other text."
            ),
            memory_file="router_memory.json"
        )

    def route(self, user_text: str) -> str:
        msg = self.agent._openai_call([
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": user_text}
        ])
        label = (msg.content or "").strip()

        if label not in self.labels:
            text = user_text.lower()
            if any(k in text for k in ["sccm", "sup", "adr", "dp", "wsus", "kb", "windows update"]):
                return "infra"
            if any(k in text for k in ["sql", "java", "c#", "selenium", "code", "stacktrace"]):
                return "coding"
            return "general"
        return label


# ----------- CREATE SUB-AGENTS -----------
infra_agent = Agent(
    name="infra",
    system_prompt="You are an IT Infra specialist. Be concise and accurate.",
    tools={"http_get": http_get},
    memory_file="infra_memory.json"
)

coding_agent = Agent(
    name="coding",
    system_prompt="You are a coding helper. Provide clean, runnable code only when relevant.",
    tools={"echo": echo},
    memory_file="coding_memory.json"
)

general_agent = Agent(
    name="general",
    system_prompt="You are a simple fast-response general assistant.",
    memory_file="general_memory.json"
)

AGENTS = {
    "infra": infra_agent,
    "coding": coding_agent,
    "general": general_agent
}

router = Router()


# ----------- FLASK APP -----------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>Multi-Agent Chatbot</title></head>
<body>
<h2>Multi-Agent Chatbot</h2>
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

chat_history = []

def load_ui_history():
    for label, agent in AGENTS.items():
        for entry in agent.memory:
            # Label memory properly so UI shows it
            role = "You" if entry["role"] == "user" else f"[{label} Agent]"
            chat_history.append({"role": role, "content": entry["content"]})

load_ui_history()


@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            chat_history.append({"role": "You", "content": user_input})

            label = router.route(user_input)
            agent = AGENTS[label]
            reply = agent.handle(user_input)
            chat_history.append({"role": f"[{label} Agent]", "content": reply})

    return render_template_string(HTML_PAGE, history=chat_history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
