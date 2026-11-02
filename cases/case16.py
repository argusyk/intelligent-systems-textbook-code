# Консультант для вступників: пошук у базі FAQ (етап Retrieval) + опційний виклик LLM
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ = [
    ("Які документи потрібні для вступу?", "Паспорт, ідентифікаційний код, документ про освіту, сертифікати НМТ."),
    ("Чи є гуртожиток?", "Так, місця надаються іногороднім здобувачам за рейтингом поселення."),
    ("Як обрати вибіркову дисципліну?", "Через особистий кабінет у період вибору, за каталогом дисциплін."),
    ("Коли починається навчання?", "Навчальний рік розпочинається 1 вересня."),
    ("Чи можна перевестися на іншу спеціальність?", "Так, після першого курсу за наявності вільних місць."),
]
vec = TfidfVectorizer().fit([q + " " + a for q, a in FAQ])
M = vec.transform([q + " " + a for q, a in FAQ])

def retrieve(question, k=2, threshold=0.1):
    sims = cosine_similarity(vec.transform([question]), M)[0]
    return [(FAQ[i], round(float(sims[i]), 2)) for i in sims.argsort()[::-1][:k] if sims[i] >= threshold]

for q in ["які документи подавати", "де жити під час навчання", "яка погода в Києві"]:
    hits = retrieve(q)
    print(q, "->", hits if hits else "немає відповіді в базі: передати оператору")

if os.environ.get("ANTHROPIC_API_KEY"):                       # генерація лише з ключем
    import anthropic
    ctx = "\n".join(a for (_, a), _ in retrieve("які документи подавати"))
    msg = anthropic.Anthropic().messages.create(
        model="claude-opus-4-8", max_tokens=300,
        system="Відповідай лише на основі наданого контексту. Якщо відповіді немає, так і скажи.",
        messages=[{"role": "user", "content": f"Контекст:\n{ctx}\n\nПитання: які документи подавати?"}])
    print(msg.content[0].text)
