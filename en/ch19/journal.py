# Processing maintenance log records: rule-based fact extraction and classification by ATA chapter
import re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

RECORDS = [
    ("12.03.2026 UR-PSA APU: EGT 705 °C at start, start time 52 s. Nozzles flushed P/N 3800123-4.", "49"),
    ("14.03.2026 UR-PSB APU does not start, starter current 0 A. Starter replaced P/N 3505880-2.", "49"),
    ("15.03.2026 UR-PSA Oil leak from the APU, oil pressure 2.1 bar. Seal replaced.", "49"),
    ("16.03.2026 UR-PSC Increased APU vibration 7.8 mm/s. Removed for bearing replacement.", "49"),
    ("18.03.2026 UR-PSB APU speed fluctuation 3 %, fuel control unit replaced P/N 4952-11.", "49"),
    ("19.03.2026 UR-PSD BLEED FAULT warning, bleed air valve checked.", "36"),
    ("20.03.2026 UR-PSA Pneumatic system pressure 28 psi below normal, PRV valve replaced.", "36"),
    ("21.03.2026 UR-PSC Air leak in the bleed duct, clamp replaced.", "36"),
    ("22.03.2026 UR-PSD Pneumatic crossfeed valve failure.", "36"),
    ("23.03.2026 UR-PSB Pressure sensor in the pneumatic line gives unstable readings.", "36"),
    ("24.03.2026 UR-PSA Left engine fuel flow 2600 kg/h above normal.", "73"),
    ("25.03.2026 UR-PSC Engine fuel filter clogged, filter replaced.", "73"),
    ("26.03.2026 UR-PSD Fuel flow sensor error, sensor replaced P/N 7710-3.", "73"),
    ("27.03.2026 UR-PSB Uneven fuel supply on engine 2, engine nozzles flushed.", "73"),
    ("28.03.2026 UR-PSA FUEL FILTER CLOG warning, filter replaced.", "73"),
    ("29.03.2026 UR-PSD Short circuit in the generator power circuit, relay replaced.", "24"),
    ("30.03.2026 UR-PSC APU generator does not come online, voltage 0 V.", "24"),
    ("31.03.2026 UR-PSB Battery discharged to 20 V, battery replaced.", "24"),
    ("01.04.2026 UR-PSA Cabin lighting flickering, power bus checked.", "24"),
    ("02.04.2026 UR-PSD Transformer rectifier unit TRU 2 failure.", "24"),
]

PATTERNS = {   # extraction rules: regular expressions
    "date": r"\b\d{2}\.\d{2}\.\d{4}\b",
    "aircraft": r"\bUR-[A-Z]{3}\b",
    "P/N": r"P/N\s*([\w-]+)",
    "EGT, °C": r"EGT\s*(\d+)\s*°C",
    "pressure, bar": r"pressure[^\d]{0,20}(\d+\.\d+)\s*bar",
    "vibration, mm/s": r"vibration[^\d]{0,10}(\d+\.\d+)\s*mm/s",
}

def extract(text):
    out = {}
    for k, p in PATTERNS.items():
        m = re.search(p, text, flags=re.I)
        if m:
            out[k] = m.group(1) if m.groups() else m.group(0)
    return out

for text, _ in RECORDS[:4]:
    print(extract(text))

texts, labels = zip(*RECORDS)
clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)), LogisticRegression(max_iter=2000))
acc = cross_val_score(clf, texts, labels, cv=5)
print(f"classification accuracy by ATA chapter (5-fold cross-validation): {acc.mean():.2f} +- {acc.std():.2f}")
print("per fold:", acc.round(2))
clf.fit(texts, labels)
for new in ["APU shut down, EGT 720 °C", "Bleed air valve does not open", "Engine 1 fuel pump replaced"]:
    print(f"{clf.predict([new])[0]}  <- {new}")
