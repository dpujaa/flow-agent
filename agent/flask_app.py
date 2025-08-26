#!/usr/bin/env python3
import os, json, io, requests, pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: OPENAI_API_KEY not found. Put it in .env or your environment.")
client = OpenAI(api_key=api_key)

# === Import shared stuff from agent.py ===
from agent import fetch_url, analyze_csv, CUSTOM_TOOLS, FUNCTION_SCHEMAS, run_with_tools, extract_text

# Flask app
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Task Agent</title>
<style>
  body { font-family: system-ui, sans-serif; background:#0b0b0c; color:#eaeaea; }
  header { padding:20px; border-bottom:1px solid #222; }
  .container { max-width:800px; margin:20px auto; }
  textarea { width:100%; min-height:140px; padding:10px; }
  button { margin-top:10px; padding:8px 14px; }
  .result { white-space:pre-wrap; margin-top:20px; padding:10px; border:1px solid #333; }
</style>
</head>
<body>
<header><h1>AI Task Agent</h1></header>
<main class="container">
  <textarea id="prompt" placeholder="Enter a task prompt..."></textarea>
  <br>
  <button onclick="runTask()">Run</button>
  <div id="status"></div>
  <div id="result" class="result"></div>
</main>
<script>
async function runTask() {
  const prompt = document.getElementById('prompt').value;
  document.getElementById('status').textContent = 'Running...';
  document.getElementById('result').textContent = '';
  const res = await fetch('/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt})
  });
  const data = await res.json();
  if (data.ok) {
    document.getElementById('result').textContent = data.text;
    document.getElementById('status').textContent = 'Done';
  } else {
    document.getElementById('result').textContent = 'Error: ' + data.error;
    document.getElementById('status').textContent = 'Failed';
  }
}
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.post("/run")
def run_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "Missing prompt"}), 400
    try:
        tools_config = [{"type": "web_search"}, *FUNCTION_SCHEMAS]
        resp = run_with_tools(prompt, tools_config)
        text = extract_text(resp)
        return jsonify({"ok": True, "text": text or ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
