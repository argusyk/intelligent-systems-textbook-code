# Генерація з доповненням пошуком (RAG): фрагментування, щільні подання, гібридний пошук,
# оцінювання пошуку (recall@k, MRR), запит з посиланнями на джерела та перевірка обґрунтованості.
# Щільні подання будуються латентно-семантичним аналізом (TF-IDF + SVD), тож програма працює
# без завантаження нейромережевих моделей; у промисловій системі їх замінюють нейромережевими
# ембедингами, а масив – векторною базою даних.
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

MANUAL = {  # навчальне «керівництво з ТО ДСУ»: розділ -> текст
 "49-10": "Загальний опис. Допоміжна силова установка забезпечує літак стисненим повітрям і електроенергією на землі. "
          "Установка складається з компресора, камери згоряння, турбіни та коробки приводів. "
          "Керування здійснює електронний блок, що контролює частоту обертання та температуру вихлопних газів.",
 "49-20": "Паливна система. Паливо подається насосом через фільтр і дозувальний блок до форсунок камери згоряння. "
          "Забруднення форсунок погіршує розпилення палива, що спричиняє затяжний запуск і підвищену температуру "
          "вихлопних газів. Підвищена витрата палива за нормальних обертів також вказує на форсунки.",
 "49-30": "Промивка форсунок. Промивку виконують на крилі без зняття установки. Тривалість процедури близько 4 год. "
          "Після промивки виконують пробний запуск і контролюють час виходу на режим та температуру вихлопних газів. "
          "Протягом 50 год напрацювання після промивки гіпотеза забруднення форсунок малоймовірна.",
 "49-40": "Система запуску. Стартер-генератор розкручує ротор до частоти самопідтримання. Нормальний час запуску "
          "не перевищує 45 с. Затяжний запуск за нормальної температури вихлопних газів вказує на стартер "
          "або на недостатню напругу бортової мережі.",
 "49-50": "Масляна система. Нормальний тиск масла на робочому режимі становить від 3,0 до 3,8 бар. "
          "Зниження тиску нижче 2,5 бар вимагає негайної зупинки установки. Причиною може бути витік, "
          "несправність насоса або засмічення фільтра. Рівень масла перевіряють після кожних 50 год.",
 "49-60": "Вібрація. Допустимий рівень вібрації становить до 4,5 мм/с. Зростання вібрації разом із падінням тиску "
          "масла свідчить про знос підшипника ротора. Установку з вібрацією понад 7 мм/с знімають з експлуатації.",
 "49-70": "Температура вихлопних газів. Гранична температура на робочому режимі становить 680 °C, під час запуску "
          "допускається короткочасне перевищення до 750 °C. Причинами перегріву бувають забруднені форсунки, "
          "знос турбіни або несправна термопара. Показання термопари порівнюють з резервним датчиком.",
 "49-80": "Термопари. Несправна термопара дає стрибкоподібні або завищені показання температури за нормальної "
          "витрати палива. Перевірку виконують вимірюванням опору; нормальний опір ланцюга від 5 до 9 Ом.",
 "49-90": "Допуск до експлуатації. Рішення про допуск установки після ТО ухвалює лише сертифікований технік "
          "з підписом у картці-наряді. Автоматизовані системи діагностики мають дорадчий характер.",
}
TESTS = [  # запитання та розділ, що містить відповідь
 ("Скільки триває промивка форсунок?", "49-30"),
 ("Який тиск масла вважається нормальним?", "49-50"),
 ("Чому запуск затяжний, а температура нормальна?", "49-40"),
 ("Що означає висока вібрація і низький тиск масла?", "49-60"),
 ("Яка гранична температура вихлопних газів?", "49-70"),
 ("Як перевірити термопару?", "49-80"),
 ("Хто ухвалює рішення про допуск установки?", "49-90"),
 ("Які ознаки забруднення форсунок?", "49-20"),
 ("Коли установку знімають з експлуатації через вібрацію?", "49-60"),
 ("Що робити після промивки форсунок?", "49-30"),
]

def chunk(text, size, overlap):
    """Фрагменти з size речень з перекриттям overlap; до кожного додається назва розділу
    (контекстне фрагментування: без неї фрагмент «Тривалість близько 4 год» втрачає зміст)."""
    title, *sents = re.split(r"(?<=\.)\s+", text)
    step = max(size - overlap, 1)
    starts = range(0, max(len(sents) - overlap, 1), step)
    return [title + " " + " ".join(sents[i:i + size]) for i in starts]

def build(size=2, overlap=1):
    return [(sec, c) for sec, t in MANUAL.items() for c in chunk(t, size, overlap)]

def normalize(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

class Index:
    def __init__(self, chunks, dim=40):
        self.chunks = chunks
        texts = [c for _, c in chunks]
        self.words = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)
        C = self.chars.transform(texts)
        self.svd = TruncatedSVD(n_components=min(dim, C.shape[1] - 1, len(texts) - 1), random_state=0).fit(C)
        self.dense = normalize(self.svd.transform(C))                  # «ембединги» фрагментів
        self.sparse = self.words.transform(texts)
    def scores(self, q, mode):
        s_sparse = (self.sparse @ self.words.transform([q]).T).toarray().ravel()
        s_dense = self.dense @ normalize(self.svd.transform(self.chars.transform([q]))).ravel()
        if mode == "лексичний": return s_sparse
        if mode == "щільний": return s_dense
        rank = lambda s: np.argsort(np.argsort(-s))                    # злиття рангів (RRF)
        return 1 / (60 + rank(s_sparse)) + 1 / (60 + rank(s_dense))
    def search(self, q, k=3, mode="гібридний"):
        s = self.scores(q, mode)
        return [(self.chunks[i][0], self.chunks[i][1], float(s[i])) for i in np.argsort(-s)[:k]]

def evaluate(index, mode, k=3):
    hits, rr = 0, 0.0
    for q, sec in TESTS:
        found = [c[0] for c in index.search(q, k=10, mode=mode)]
        hits += sec in found[:k]
        rr += 1 / (found.index(sec) + 1) if sec in found else 0
    return hits / len(TESTS), rr / len(TESTS)

def make_prompt(q, found):
    ctx = "\n".join(f"[{sec}] {text}" for sec, text, _ in found)
    return ("Відповідай лише на основі фрагментів керівництва. Після кожного твердження вкажи розділ "
            "у квадратних дужках. Якщо відповіді немає у фрагментах, напиши «немає даних».\n\n"
            f"<фрагменти>\n{ctx}\n</фрагменти>\n\nЗапитання: {q}")

def offline_answer(q, found, n=3):
    """Імітація моделі: n найрелевантніших речень знайдених фрагментів з посиланнями на розділи."""
    qw = {w[:5] for w in re.findall(r"\w{4,}", q.lower())}           # грубе врахування словоформ
    cand = []
    for sec, text, _ in found:
        for sent in re.split(r"(?<=\.)\s+", text)[1:]:              # без назви розділу
            cand.append((len(qw & {w[:5] for w in re.findall(r"\w{4,}", sent.lower())}), sent, sec))
    best, seen = [], set()
    for sc, sent, sec in sorted(cand, key=lambda c: -c[0]):
        if sent not in seen and len(best) < n:
            best.append(f"{sent} [{sec}]"); seen.add(sent)
    return " ".join(best)

def grounded(answer, found):
    """Частка змістовних слів відповіді, що є у знайдених фрагментах, і коректність посилань."""
    ctx = " ".join(t for _, t, _ in found).lower()
    words = [w for w in re.findall(r"\w{4,}", re.sub(r"\[.*?\]", "", answer).lower())]
    cites = re.findall(r"\[(\d\d-\d\d)\]", answer)
    return sum(w in ctx for w in words) / max(len(words), 1), all(c in {s for s, _, _ in found} for c in cites)

if __name__ == "__main__":
    print("фрагментів | режим      | recall@3 | MRR")
    results = {}
    for size, ov in [(1, 0), (2, 1), (4, 1)]:
        idx = Index(build(size, ov))
        for mode in ["лексичний", "щільний", "гібридний"]:
            r, m = evaluate(idx, mode); results[(size, mode)] = (r, m)
            print(f"{size} реч. ({len(idx.chunks):2d}) | {mode:10s} | {r:.2f}     | {m:.2f}")
    idx = Index(build(2, 1))
    q = "Скільки триває промивка форсунок і що робити після неї?"
    found = idx.search(q, k=3)
    print("\n" + make_prompt(q, found))
    ans = offline_answer(q, found)
    share, cites_ok = grounded(ans, found)
    print("\nВідповідь:", ans)
    print(f"обґрунтованість: {share:.2f}; посилання коректні: {cites_ok}")
    bad = "Промивка триває 30 хв і після неї установку можна одразу допускати до польоту [49-30]."
    print("вигадана відповідь: обґрунтованість %.2f" % grounded(bad, found)[0])
