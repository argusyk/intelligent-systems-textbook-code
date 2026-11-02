# Classifying bank customer requests with a language model, validating the JSON response
# and retrying on a format error. Without ANTHROPIC_API_KEY an offline imitation is used.
import json, os, re

CATS = ["card", "loan", "transfer", "app", "other"]
EXAMPLES = [("My card was blocked after three failed PIN attempts", "card"),
            ("I want to repay my loan early", "loan")]
TESTS = [("Card payment online does not go through", "card"),
         ("The money did not reach my friend's account", "transfer"),
         ("The app crashes after the update",
          "app"), ("What is the rate on a consumer loan?", "loan"),
         ("I lost my card, what should I do?",
          "card"), ("A transfer by bank details has been pending for three days", "transfer"),
         ("I cannot log in to the app, it shows an error",
          "app"), ("Where is the nearest branch?", "other"),
         ("Can I change the payment schedule of my loan?",
          "loan"), ("Fee for a transfer to another bank's card", "transfer")]

def prompt(text):
    shots = "\n".join(f'Request: {t}\nAnswer: {{"category": "{c}", "urgent": false}}'
                      for t, c in EXAMPLES)
    return (f"Assign the bank customer request to one of the categories {CATS}. Set "
            f"urgent=true "
            f"if the customer may lose money. Answer with JSON "
            f"only.\n\n{shots}\n\nRequest: {text}\nAnswer:")

def offline_model(p, attempt):
    """Model imitation: keywords; on the first attempt it sometimes breaks the format."""
    text = p.rsplit("Request: ", 1)[1].lower()
    rules = [("card", "card"), ("loan", "loan"), ("repay", "loan"), ("transfer",
                                                                     "transfer"),
             ("money", "transfer"), ("app", "app"), ("log in", "app")]
    cat = next((c for k, c in rules if k in text), "other")
    urgent = any(w in text for w in ["lost", "did not reach", "block"])
    if attempt == 0 and "?" in text:  # a typical error: text around the JSON
        flag = str(urgent).lower()
        return f'Sure! Here is the answer: {{"category": "{cat}", "urgent": {flag}}}'
    return json.dumps({"category": cat, "urgent": urgent})

def call(p, attempt):
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic().messages.create(model="claude-opus-4-8", max_tokens=100,
                                                  messages=[{"role": "user", "content": p}])
        return r.content[0].text
    return offline_model(p, attempt)

def validate(raw):
    d = json.loads(raw)                                           # ValueError if not JSON
    if d.get("category") not in CATS or not isinstance(d.get("urgent"), bool):
        raise ValueError("schema violated")
    return d

stats = {"correct": 0, "retries": 0, "failures": 0}
for text, gold in TESTS:
    for attempt in range(3):
        try:
            d = validate(call(prompt(text), attempt)); break
        except ValueError:
            stats["retries"] += 1
    else:
        stats["failures"] += 1; continue
    stats["correct"] += d["category"] == gold
print(f"accuracy {stats['correct']}/{len(TESTS)}, retries {stats['retries']}, "
      f"failures {stats['failures']}")
