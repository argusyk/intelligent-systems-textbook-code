# Retrieval-augmented generation (RAG): chunking, dense representations, hybrid search,
# retrieval evaluation (recall@k, MRR), a prompt with source citations and a groundedness check.
# Dense representations are built by latent semantic analysis (TF-IDF + SVD), so the program runs
# without downloading neural models; in an industrial system they are replaced by neural
# embeddings, and the array by a vector database.
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

MANUAL = {  # a training "APU maintenance manual": section -> text
 "49-10": "General description. The auxiliary power unit supplies the aircraft with compressed air and electrical power on the ground. "
          "The unit consists of a compressor, a combustion chamber, a turbine and an accessory gearbox. "
          "It is controlled by an electronic control unit that monitors the rotational speed and the exhaust gas temperature.",
 "49-20": "Fuel system. Fuel is delivered by a pump through a filter and a metering unit to the combustion chamber nozzles. "
          "Nozzle fouling degrades fuel atomisation, which causes a slow start and an elevated exhaust gas "
          "temperature. Increased fuel flow at normal speed also points to the nozzles.",
 "49-30": "Nozzle flushing. Flushing is performed on the wing without removing the unit. The procedure takes about 4 h. "
          "After flushing, a test start is performed and the time to reach the operating mode and the exhaust gas temperature are monitored. "
          "For 50 operating hours after flushing, the nozzle fouling hypothesis is unlikely.",
 "49-40": "Starting system. The starter-generator spins the rotor up to the self-sustaining speed. The normal start time "
          "does not exceed 45 s. A slow start with a normal exhaust gas temperature points to the starter "
          "or to insufficient voltage of the aircraft electrical network.",
 "49-50": "Oil system. The normal oil pressure in the operating mode is from 3.0 to 3.8 bar. "
          "A pressure drop below 2.5 bar requires an immediate shutdown of the unit. The cause may be a leak, "
          "a pump failure or a clogged filter. The oil level is checked every 50 h.",
 "49-60": "Vibration. The permissible vibration level is up to 4.5 mm/s. Rising vibration together with falling oil "
          "pressure indicates wear of the rotor bearing. A unit with vibration above 7 mm/s is removed from service.",
 "49-70": "Exhaust gas temperature. The limit temperature in the operating mode is 680 °C; during a start "
          "a short-term excess up to 750 °C is allowed. Overheating may be caused by fouled nozzles, "
          "turbine wear or a faulty thermocouple. The thermocouple readings are compared with the backup sensor.",
 "49-80": "Thermocouples. A faulty thermocouple gives jumpy or overstated temperature readings at normal "
          "fuel flow. It is tested by measuring the resistance; the normal circuit resistance is 5 to 9 ohms.",
 "49-90": "Release to service. The decision to release the unit after maintenance is made only by a certified technician "
          "who signs the work order card. Automated diagnostic systems are advisory.",
}
TESTS = [  # question and the section that contains the answer
 ("How long does nozzle flushing take?", "49-30"),
 ("What oil pressure is considered normal?", "49-50"),
 ("Why is the start slow while the temperature is normal?", "49-40"),
 ("What do high vibration and low oil pressure mean?", "49-60"),
 ("What is the exhaust gas temperature limit?", "49-70"),
 ("How do I check a thermocouple?", "49-80"),
 ("Who decides on releasing the unit?", "49-90"),
 ("What are the signs of nozzle fouling?", "49-20"),
 ("When is a unit removed from service because of vibration?", "49-60"),
 ("What should be done after nozzle flushing?", "49-30"),
]

def chunk(text, size, overlap):
    """Chunks of size sentences with overlap; the section title is prepended to each
    (contextual chunking: without it the chunk "The procedure takes about 4 h" loses its meaning)."""
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
        self.dense = normalize(self.svd.transform(C))                  # chunk "embeddings"
        self.sparse = self.words.transform(texts)
    def scores(self, q, mode):
        s_sparse = (self.sparse @ self.words.transform([q]).T).toarray().ravel()
        s_dense = self.dense @ normalize(self.svd.transform(self.chars.transform([q]))).ravel()
        if mode == "lexical": return s_sparse
        if mode == "dense": return s_dense
        rank = lambda s: np.argsort(np.argsort(-s))                    # reciprocal rank fusion (RRF)
        return 1 / (60 + rank(s_sparse)) + 1 / (60 + rank(s_dense))
    def search(self, q, k=3, mode="hybrid"):
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
    return ("Answer only on the basis of the manual fragments. After each statement give the section "
            "in square brackets. If the answer is not in the fragments, write 'no data'.\n\n"
            f"<fragments>\n{ctx}\n</fragments>\n\nQuestion: {q}")

def offline_answer(q, found, n=3):
    """Model imitation: the n most relevant sentences of the retrieved chunks with section citations."""
    qw = {w[:5] for w in re.findall(r"\w{4,}", q.lower())}           # a crude way of handling word forms
    cand = []
    for sec, text, _ in found:
        for sent in re.split(r"(?<=\.)\s+", text)[1:]:              # without the section title
            cand.append((len(qw & {w[:5] for w in re.findall(r"\w{4,}", sent.lower())}), sent, sec))
    best, seen = [], set()
    for sc, sent, sec in sorted(cand, key=lambda c: -c[0]):
        if sent not in seen and len(best) < n:
            best.append(f"{sent} [{sec}]"); seen.add(sent)
    return " ".join(best)

def grounded(answer, found):
    """Share of the answer's content words found in the retrieved chunks, and citation validity."""
    ctx = " ".join(t for _, t, _ in found).lower()
    words = [w for w in re.findall(r"\w{4,}", re.sub(r"\[.*?\]", "", answer).lower())]
    cites = re.findall(r"\[(\d\d-\d\d)\]", answer)
    return sum(w in ctx for w in words) / max(len(words), 1), all(c in {s for s, _, _ in found} for c in cites)

if __name__ == "__main__":
    print("chunks        | mode       | recall@3 | MRR")
    results = {}
    for size, ov in [(1, 0), (2, 1), (4, 1)]:
        idx = Index(build(size, ov))
        for mode in ["lexical", "dense", "hybrid"]:
            r, m = evaluate(idx, mode); results[(size, mode)] = (r, m)
            print(f"{size} sent. ({len(idx.chunks):2d}) | {mode:10s} | {r:.2f}     | {m:.2f}")
    idx = Index(build(2, 1))
    q = "How long does nozzle flushing take and what should be done after it?"
    found = idx.search(q, k=3)
    print("\n" + make_prompt(q, found))
    ans = offline_answer(q, found)
    share, cites_ok = grounded(ans, found)
    print("\nAnswer:", ans)
    print(f"groundedness: {share:.2f}; citations valid: {cites_ok}")
    bad = "Flushing takes 30 minutes, and the aircraft may depart immediately afterwards [49-30]."
    print("fabricated answer: groundedness %.2f" % grounded(bad, found)[0])
