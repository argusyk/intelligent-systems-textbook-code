# Класифікація звернень клієнтів банку мовною моделлю з перевіркою JSON-відповіді
# і повторним запитом у разі помилки формату. Без ANTHROPIC_API_KEY працює офлайн-імітація.
import json, os, re

CATS = ["картка", "кредит", "переказ", "застосунок", "інше"]
EXAMPLES = [("Картку заблокували після трьох невдалих спроб PIN", "картка"),
            ("Хочу достроково погасити позику", "кредит")]
TESTS = [("Не проходить оплата карткою в інтернеті", "картка"),
         ("Гроші на рахунок друга не дійшли", "переказ"),
         ("Застосунок вилітає після оновлення",
          "застосунок"), ("Яка ставка на споживчий кредит?", "кредит"),
         ("Втратив картку, що робити?",
          "картка"), ("Переказ за реквізитами висить третій день", "переказ"),
         ("Не можу увійти в застосунок, пише помилку",
          "застосунок"), ("Де найближче відділення?", "інше"),
         ("Чи можна змінити графік платежів по кредиту?",
          "кредит"), ("Комісія за переказ на картку іншого банку", "переказ")]

def prompt(text):
    shots = "\n".join(f'Звернення: {t}\nВідповідь: {{"category": "{c}", "urgent": false}}'
                      for t, c in EXAMPLES)
    return (f"Віднеси звернення клієнта банку до однієї з категорій {CATS}. Познач "
            f"urgent=true, "
            f"якщо клієнт може втратити гроші. Відповідай лише "
            f"JSON.\n\n{shots}\n\nЗвернення: {text}\nВідповідь:")

def offline_model(p, attempt):
    """Імітація моделі: ключові слова; на першій спробі іноді порушує формат."""
    text = p.rsplit("Звернення: ", 1)[1].lower()
    rules = [("картк", "картка"), ("кредит", "кредит"), ("позик", "кредит"), ("переказ",
                                                                              "переказ"),
             ("гроші", "переказ"), ("застосун", "застосунок"), ("увійти", "застосунок")]
    cat = next((c for k, c in rules if k in text), "інше")
    urgent = any(w in text for w in ["втратив", "не дійшли", "заблок"])
    if attempt == 0 and "?" in text:  # типова помилка: текст навколо JSON
        flag = str(urgent).lower()
        return f'Звісно! Ось відповідь: {{"category": "{cat}", "urgent": {flag}}}'
    return json.dumps({"category": cat, "urgent": urgent}, ensure_ascii=False)

def call(p, attempt):
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic().messages.create(model="claude-opus-4-8", max_tokens=100,
                                                  messages=[{"role": "user", "content": p}])
        return r.content[0].text
    return offline_model(p, attempt)

def validate(raw):
    d = json.loads(raw)                                           # ValueError, якщо не JSON
    if d.get("category") not in CATS or not isinstance(d.get("urgent"), bool):
        raise ValueError("порушено схему")
    return d

stats = {"правильно": 0, "повторів": 0, "відмов": 0}
for text, gold in TESTS:
    for attempt in range(3):
        try:
            d = validate(call(prompt(text), attempt)); break
        except ValueError:
            stats["повторів"] += 1
    else:
        stats["відмов"] += 1; continue
    stats["правильно"] += d["category"] == gold
print(f"точність {stats['правильно']}/{len(TESTS)}, повторних запитів {stats['повторів']}, "
      f"відмов {stats['відмов']}")
