# Мультиагентна схема «оркестратор – виконавці – критик» для «ДСУ-Експерта».
# Кожен агент має одну роль і мінімальні права; обмін іде структурованими повідомленнями.
# Офлайн агенти-виконавці – звичайні функції; у реальній системі діагност і бібліотекар
# можуть бути окремими викликами мовної моделі зі своїми інструментами (agent.py, rag.py).
import re
from agent import get_parameters, run_diagnosis
from rag import Index, build

INDEX = Index(build(2, 1))
LOG = []

def send(frm, to, kind, body):
    LOG.append((frm, to, kind, body))
    return body

def diagnostician(task):                       # права: читання параметрів, запуск ЕС
    p = get_parameters(task["unit"])
    return {"params": p, **run_diagnosis(p)}

def librarian(task):                           # права: лише пошук у керівництві
    return [(sec, text) for sec, text, _ in INDEX.search(task["query"], k=2)]

def critic(draft, evidence):                   # права: лише читання чернетки й доказів
    problems = []
    if not re.search(r"\[\d\d-\d\d\]", draft): problems.append("немає посилання на керівництво")
    if re.search(r"допуск\w*\s+(до польоту\s+)?(можна|дозволено)", draft.lower()):
        problems.append("агент не може надавати допуск")
    if "CF" not in draft: problems.append("немає коефіцієнта впевненості")
    cited = set(re.findall(r"\[(\d\d-\d\d)\]", draft))
    if not cited <= {s for s, _ in evidence}: problems.append("посилання на незнайдений розділ")
    return problems

def orchestrator(unit, question, max_rounds=2):
    d = send("оркестратор", "діагност", "завдання", {"unit": unit})
    diag = send("діагност", "оркестратор", "результат", diagnostician(d))
    q = send("оркестратор", "бібліотекар", "завдання", {"query": "промивка форсунок процедура тривалість"})
    ev = send("бібліотекар", "оркестратор", "результат", librarian(q))
    draft = (f"Установка {unit}: ймовірна причина – забруднені форсунки, "
             f"CF = {diag['H1_nozzles']} (правила {', '.join(diag['rules'])}). "
             "Параметри в нормі для інших систем, допуск можна надати.")   # навмисна помилка
    for rnd in range(max_rounds):
        issues = send("критик", "оркестратор", "зауваження", critic(draft, ev))
        if not issues:
            return draft
        draft = (f"Установка {unit}: ймовірна причина – забруднені форсунки, "
                 f"CF = {diag['H1_nozzles']} (правила {', '.join(diag['rules'])}). "
                 f"Рекомендовано промивку на крилі, близько 4 год [{ev[0][0]}]. "
                 "Рішення про допуск ухвалює сертифікований технік [49-90].")
        ev = ev + [(s, t) for s, t, _ in INDEX.search("допуск установки рішення технік", k=1)]
    return draft + " (критик має зауваження – потрібна перевірка людиною)"

if __name__ == "__main__":
    answer = orchestrator("UR-PSA", "Чому затяжний запуск і що робити?")
    for frm, to, kind, body in LOG:
        text = str(body)
        print(f"{frm:>11s} -> {to:<11s} {kind:10s} {text[:70]}{'...' if len(text) > 70 else ''}")
    print("\nПідсумок:", answer)
