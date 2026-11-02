# Нормалізація українського тексту та її вплив на пошук: апострофи, латинські «двійники»
# кириличних літер, регістр, грубе врахування словоформ (усічення до основи)
import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch17"))
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rag import MANUAL, TESTS

APOSTROPHES = "'’ʼ`´‘"
HOMOGLYPHS = str.maketrans("aceiopxyAСEHIKMOPTXB", "асеіорхуАСЕНІКМОРТХВ")   # латиниця -> кирилиця

def normalize(text):
    text = re.sub(f"[{APOSTROPHES}]", "'", text)                    # один вид апострофа
    # латинські літери всередині кириличних слів замінюємо кириличними двійниками
    text = re.sub(r"\w+", lambda m: m.group(0).translate(HOMOGLYPHS)
                  if re.search("[а-яіїєґА-ЯІЇЄҐ]", m.group(0)) else m.group(0), text)
    return text.lower()

def stem(word, n=5):
    """Грубе врахування словоформ: перші n літер (термопару, термопари -> «термо»)."""
    return word[:n] if len(word) > n else word

docs = [(sec, text) for sec, text in MANUAL.items()]
texts = [t for _, t in docs]
variants = {
    "слова без нормалізації": dict(analyzer="word", lowercase=False),
    "нормалізація": dict(analyzer="word", preprocessor=normalize),
    "нормалізація + основи": dict(analyzer=lambda t: [stem(w) for w in re.findall(r"\w+", normalize(t))]),
    "символьні n-грами": dict(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize),
}
queries = [(q, s) for q, s in TESTS] + [("Скільки тpиває пpомивка?", "49-30"),      # латинські «p»
                                        ("Опір термопари", "49-80"), ("ТИСК МАСЛА", "49-50")]
print(f"{'подання':26s} recall@1 (з {len(queries)} запитань)")
for name, kw in variants.items():
    vec = TfidfVectorizer(**kw).fit(texts)
    D = vec.transform(texts)
    hits = sum(docs[int(np.argmax((D @ vec.transform([q]).T).toarray()))][0] == s for q, s in queries)
    print(f"{name:26s} {hits / len(queries):.2f}")
print(normalize("Скільки тpиває пpомивка форсунок, м’ята ʼінструкціяʼ?"))
