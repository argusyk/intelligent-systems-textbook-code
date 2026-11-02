# Admissions assistant: FAQ retrieval (the Retrieval stage) + an optional LLM call
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ = [
    ("What documents are required for admission?",
     "Passport, taxpayer ID, education certificate, NMT test certificates."),
    ("Is there a dormitory?",
     "Yes, places are provided to out-of-town students according to the housing ranking."),
    ("How do I choose an elective course?",
     "Through your personal account during the selection period, using the course catalogue."),
    ("When do classes start?", "The academic year begins on 1 September."),
    ("Can I transfer to another programme?",
     "Yes, after the first year if there are vacant places."),
]
vec = TfidfVectorizer(stop_words="english").fit([q + " " + a for q, a in FAQ])
M = vec.transform([q + " " + a for q, a in FAQ])

def retrieve(question, k=2, threshold=0.1):
    sims = cosine_similarity(vec.transform([question]), M)[0]
    return [(FAQ[i], round(float(sims[i]),
                           2)) for i in sims.argsort()[::-1][:k] if sims[i] >= threshold]

for q in ["which documents do I need for admission", "where to live during the first year",
          "what is the weather in Kyiv"]:
    hits = retrieve(q)
    print(q, "->", hits if hits else "no answer in the base: forward to an operator")

if os.environ.get("ANTHROPIC_API_KEY"):                       # generation only with a key
    import anthropic
    ctx = "\n".join(a for (_, a), _ in retrieve("which documents do I need for admission"))
    msg = anthropic.Anthropic().messages.create(
        model="claude-opus-4-8", max_tokens=300,
        system="Answer only on the basis of the context provided. "
               "If there is no answer, say so.",
        messages=[{"role": "user",
                   "content": f"Context:\n{ctx}\n\nQuestion: which documents do I need for admission?"}])
    print("".join(b.text for b in msg.content if b.type == "text"))
