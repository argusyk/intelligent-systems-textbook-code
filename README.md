# Супровідний код посібника «Інтелектуальні системи»

Код до навчального посібника: Дорогий Я. Ю. Інтелектуальні системи : електронний навчальний посібник. Київ : КАІ, 2026.
Каталог `code/`, на який посилається текст посібника, відповідає кореню цього репозиторію
(наприклад, `code/ch17/agent.py` → [`ch17/agent.py`](ch17/agent.py)).

Companion code of the textbook *Intelligent Systems* (Kyiv Aviation Institute, 2026). The English edition of the code
(comments, strings and data translated) is in [`en/`](en/); the `code/` directory referred to in the English textbook
corresponds to that folder.

## Як запустити

```
pip install -r requirements.txt
python cases/case07.py
```

- `chNN/listing_KK.*` – лістинги з тексту розділу NN у порядку появи (`.py` – Python, `.clp` – CLIPS, `.pl` – Prolog, `.ttl` – Turtle).
- `labs/` – фрагменти з прикладів виконання лабораторних робіт (listing_04 і listing_05 доповнюють код розділів 12 і 13 і окремо не запускаються).
- `cases/caseNN.py` – практичні кейси розділів (підрозділ «Практичний кейс»).
- `_report.txt` – результат автоматичного запуску всіх лістингів (`py build/extract_code.py`).

Лістинги Prolog перевіряйте в SWI-Prolog (https://www.swi-prolog.org). Приклади з Claude API
виконуються лише за наявності змінної середовища `ANTHROPIC_API_KEY`.

- `chNN/*.py` для розділів 17–26 – повні програми, фрагменти яких наведено в тексті.
- `data/` – завантажувачі відкритих наборів даних NASA C-MAPSS (`cmapss.py`) і MovieLens 100K (`movielens.py`);
  самі набори завантажте вручну з сайтів NASA і GroupLens, без них програми працюють на демонстраційних файлах.
- `tests/tests_moodle.gift` – тести для самоконтролю у форматі GIFT для імпорту в банк запитань Moodle.
