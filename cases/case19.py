# Видобування реквізитів з рахунків на оплату:
# регулярні вирази + перевірка контрольної суми IBAN
import re

def iban_ok(iban):
    """Перевірка IBAN за ISO 13616: 4 символи в кінець, літери -> числа, остача від 97 = 1."""
    s = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in s)) % 97 == 1

def make_iban(mfo, account):  # генерація коректного IBAN для прикладів
    base = mfo + account.zfill(19)
    check = 98 - int("".join(str(int(ch, 36)) for ch in base + "UA00")) % 97
    return f"UA{check:02d}{base}"

good, bad = make_iban("305299", "26007012345678"), make_iban("322001", "26000055501234")
bad = bad[:10] + ("1" if bad[10] != "1" else "2") + bad[11:]  # одна помилка набору
DOCS = [f"Рахунок № 118 від 03.09.2026. Постачальник ТОВ «Авіасервіс», код ЄДРПОУ 38457120, "
        f"IBAN {good}. До сплати: 12 450,00 грн, у т.ч. ПДВ 2 075,00 грн.",
        f"РАХУНОК-ФАКТУРА №2026/77 від 15.09.2026 р. ФОП Іваненко О.П., РНОКПП 3012456789, "
        f"р/р {bad}. Сума до оплати 3 600,50 грн без ПДВ."]
PAT = {"номер": r"№\s*([\w/-]+)", "дата": r"від\s+(\d{2}\.\d{2}\.\d{4})",
       "ЄДРПОУ": r"ЄДРПОУ\s+(\d{8})\b", "РНОКПП": r"РНОКПП\s+(\d{10})\b",
       "IBAN": r"\b(UA\d{27})\b", "сума": r"(?:сплати|оплати):?\s*([\d\s]+,\d{2})\s*грн"}
for doc in DOCS:
    rec = {k: (m.group(1).strip() if (m := re.search(p, doc)) else None) for k,
           p in PAT.items()}
    rec["сума"] = float(rec["сума"].replace(" ", "").replace(",",
                                                             ".")) if rec["сума"] else None
    rec["IBAN коректний"] = iban_ok(rec["IBAN"]) if rec["IBAN"] else None
    print({k: v for k, v in rec.items() if v is not None})
