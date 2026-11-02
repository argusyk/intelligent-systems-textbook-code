# Extracting structured data with a language model by forcing a call of a "schema tool".
# Without an API key a rule-based fallback is used (the same result format).
import json, os, re

SCHEMA = {"name": "record", "strict": True, "description": "A structured maintenance log record",
          "input_schema": {"type": "object", "properties": {
              "aircraft": {"type": "string"}, "system": {"type": "string", "enum": ["APU", "pneumatics", "fuel", "electrical"]},
              "symptom": {"type": "string"}, "action": {"type": "string"},
              "part_number": {"type": ["string", "null"]}}, "required": ["aircraft", "system", "symptom", "action", "part_number"],
              "additionalProperties": False}}

def validate(d):
    req = SCHEMA["input_schema"]["required"]
    missing = [k for k in req if not d.get(k)]
    bad_enum = d.get("system") not in SCHEMA["input_schema"]["properties"]["system"]["enum"]
    return (not missing and not bad_enum), missing

def extract_llm(text):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model="claude-opus-4-8", max_tokens=400, tools=[SCHEMA],
        tool_choice={"type": "tool", "name": "record"},          # force an answer that follows the schema
        messages=[{"role": "user", "content": f"Extract the data from the maintenance log record:\n{text}"}])
    return next(b.input for b in r.content if b.type == "tool_use")

def extract_rules(text):
    part = re.search(r"P/N\s*([\w-]+)", text)
    sent = [s.strip() for s in re.split(r"\.\s", text) if s.strip()]
    return {"aircraft": re.search(r"UR-[A-Z]{3}", text).group(0), "system": "APU" if "APU" in text else "fuel",
            "symptom": sent[0][18:], "action": sent[-1].rstrip("."), "part_number": part.group(1) if part else None}

text = "12.03.2026 UR-PSA APU: EGT 705 °C at start, start time 52 s. Nozzles flushed P/N 3800123-4."
rec = extract_llm(text) if os.environ.get("ANTHROPIC_API_KEY") else extract_rules(text)
print(json.dumps(rec, indent=1))
print("matches the schema:", validate(rec))
