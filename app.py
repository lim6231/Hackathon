import os
from flask import Flask, request, render_template_string
from openai import OpenAI

# ----------- SAFETY CHECK -----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

client = OpenAI(api_key=api_key)

# ----------- FLASK APP -----------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head><title>AI Test Coverage Optimizer</title></head>
<body>
<h2>AI Test Coverage Optimizer</h2>
<form method="post">
<textarea name="user_input" rows="6" cols="80" placeholder="Paste user stories, requirements, log data, or defects here..."></textarea><br>
<input type="submit" value="Generate Test Plan"/>
</form>
<div style="margin-top:20px;">
{% if output %}
<h3>Prioritized Test Plan:</h3>
<pre>{{ output }}</pre>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            # ----------- CALL OPENAI -----------
            prompt = f"""
You are an AI Test Coverage Optimizer. 

Input: {user_input}

Output:
1. Risk scores per feature or module
2. Most impactful test cases
3. Missing coverage areas
4. A prioritized test plan

Return as clean, structured text, no extra commentary.
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1500
                )
                output = resp.choices[0].message.content.strip()
            except Exception as e:
                output = f"[ERROR] {e}"

    return render_template_string(HTML_PAGE, output=output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
