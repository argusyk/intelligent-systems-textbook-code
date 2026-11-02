# Видобування структурованих даних мовною моделлю з примусовим викликом «інструмента-схеми».
# Без ключа API використовується правило-орієнтований запасний варіант (той самий формат результату).
import json, os, re

SCHEMA = {"name": "record", "strict": True, "description": "Структурований запис журналу ТО",
          "input_schema": {"type": "object", "properties": {
              "aircraft": {"type": "string"}, "system": {"type": "string", "enum": ["ДСУ", "пневматика", "паливо", "електрика"]},
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
        tool_choice={"type": "tool", "name": "record"},          # змусити відповісти за схемою
        messages=[{"role": "user", "content": f"Видобудь дані із запису журналу ТО:\n{text}"}])
    return next(b.input for b in r.content if b.type == "tool_use")

def extract_rules(text):
    part = re.search(r"P/N\s*([\w-]+)", text)
    sent = [s.strip() for s in re.split(r"\.\s", text) if s.strip()]
    return {"aircraft": re.search(r"UR-[A-Z]{3}", text).group(0), "system": "ДСУ" if "ДСУ" in text else "паливо",
            "symptom": sent[0][17:], "action": sent[-1].rstrip("."), "part_number": part.group(1) if part else None}

text = "12.03.2026 UR-PSA ДСУ: EGT 705 °C при запуску, запуск 52 с. Промито форсунки P/N 3800123-4."
rec = extract_llm(text) if os.environ.get("ANTHROPIC_API_KEY") else extract_rules(text)
print(json.dumps(rec, ensure_ascii=False, indent=1))
print("відповідає схемі:", validate(rec))
