#!/usr/bin/env python3
import os, sys, io, json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found. Put it in .env or your environment.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# =======================
# Custom tool definitions
# =======================

def fetch_url(url: str, take_table: bool = True) -> dict:
    """Fetch a web page, return title, h1s, and optional first table preview."""
    resp = requests.get(url, timeout=20, headers={"User-Agent": "agent/0.1"})
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    table_preview = None
    if take_table:
        table = soup.find("table")
        if table:
            rows = []
            for tr in table.find_all("tr")[:6]:
                cells = [c.get_text(strip=True) for c in tr.find_all(["td","th"])]
                rows.append(cells)
            table_preview = rows
    return {"title": title, "h1s": h1s, "table_preview": table_preview, "length": len(html)}

def analyze_csv(csv: str = None, path: str = None) -> dict:
    """Basic data profiling: rows/cols, dtypes, describe(), head(). Provide either 'csv' or 'path'."""
    if path:
        df = pd.read_csv(path)
    elif csv:
        df = pd.read_csv(io.StringIO(csv))
    else:
        raise ValueError("Provide csv (string) or path.")
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "non_null_counts": df.count().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.to_dict().items()},
        "describe": df.describe(include="all", datetime_is_numeric=True).fillna("").to_dict(),
        "head": df.head(5).to_dict(orient="records"),
    }
    return summary

CUSTOM_TOOLS = {
    "fetch_url": fetch_url,
    "analyze_csv": analyze_csv,
}

FUNCTION_SCHEMAS = [
    {
        "type": "function",
        "name": "fetch_url",
        "description": "Fetch a URL and summarize structure; useful for scraping and quick page inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP or HTTPS URL to fetch"},
                "take_table": {"type": "boolean", "description": "Whether to return first table preview", "default": True}
            },
            "required": ["url"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "analyze_csv",
        "description": "Analyze CSV data from a file path or inline CSV content. Provide either 'csv' or 'path'.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv":  {"type": "string", "description": "Inline CSV content"},
                "path": {"type": "string", "description": "Path to a CSV file"}
            },
            "additionalProperties": False
        }
    }
]

# -----------------------
# Helpers
# -----------------------
def extract_text(resp) -> str:
    """Pull human-readable text from Responses API output blocks."""
    chunks = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") == "message" and getattr(item, "content", None):
            for block in item.content:
                if getattr(block, "type", "") in ("output_text", "text"):
                    chunks.append(getattr(block, "text", "") or getattr(block, "value", ""))
        elif getattr(item, "type", "") in ("output_text", "text"):
            chunks.append(getattr(item, "text", "") or getattr(item, "value", ""))
    return "\n".join([c for c in chunks if c])

def run_with_tools(prompt: str, tools_config):
    """Multi-round tool loop until no more tool calls."""
    resp = client.responses.create(
        model="gpt-4.1",
        input=[{"role": "user", "content": prompt}],
        tools=tools_config,
    )

    while True:
        tool_calls = [it for it in (getattr(resp, "output", []) or []) if getattr(it, "type", "") == "tool_call"]
        if not tool_calls:
            break

        tool_outputs = []
        for call in tool_calls:
            name = call.name
            args = call.arguments or {}
            if name in CUSTOM_TOOLS:
                try:
                    result = CUSTOM_TOOLS[name](**args)
                    tool_outputs.append({"tool_call_id": call.id, "output": json.dumps(result)})
                except Exception as e:
                    tool_outputs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(e)})})

        if tool_outputs:
            resp = client.responses.submit_tool_outputs(
                response_id=resp.id,
                tool_outputs=tool_outputs,
            )
        else:
            break
    return resp

# -----------------------
# CLI entrypoint
# -----------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your natural-language task here"')
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:])
    tools_config = [{"type": "web_search"}, *FUNCTION_SCHEMAS]
    resp = run_with_tools(user_prompt, tools_config)
    final_text = extract_text(resp)
    print(final_text if final_text.strip() else "[No final text output]")

if __name__ == "__main__":
    main()
