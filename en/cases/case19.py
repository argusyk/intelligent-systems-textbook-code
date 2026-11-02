# Extracting payment details from invoices:
# regular expressions + IBAN checksum validation
import re

def iban_ok(iban):
    """IBAN check per ISO 13616: 4 characters to the end, letters -> numbers, remainder mod 97 = 1."""
    s = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in s)) % 97 == 1

def make_iban(mfo, account):  # generating a valid IBAN for the examples
    base = mfo + account.zfill(19)
    check = 98 - int("".join(str(int(ch, 36)) for ch in base + "UA00")) % 97
    return f"UA{check:02d}{base}"

good, bad = make_iban("305299", "26007012345678"), make_iban("322001", "26000055501234")
bad = bad[:10] + ("1" if bad[10] != "1" else "2") + bad[11:]  # a single typing error
DOCS = [f"Invoice No. 118 dated 03.09.2026. Supplier Aviaservice LLC, EDRPOU code 38457120, "
        f"IBAN {good}. Amount due: 12 450.00 UAH, incl. VAT 2 075.00 UAH.",
        f"INVOICE No. 2026/77 dated 15.09.2026. Sole proprietor O. P. Ivanenko, "
        f"RNOKPP 3012456789, account {bad}. Amount payable 3 600.50 UAH without VAT."]
PAT = {"number": r"No\.\s*([\w/-]+)", "date": r"dated\s+(\d{2}\.\d{2}\.\d{4})",
       "EDRPOU": r"EDRPOU code\s+(\d{8})\b", "RNOKPP": r"RNOKPP\s+(\d{10})\b",
       "IBAN": r"\b(UA\d{27})\b", "amount": r"(?:due|payable):?\s*([\d\s]+\.\d{2})\s*UAH"}
for doc in DOCS:
    rec = {k: (m.group(1).strip() if (m := re.search(p, doc)) else None) for k,
           p in PAT.items()}
    rec["amount"] = float(rec["amount"].replace(" ", "")) if rec["amount"] else None
    rec["IBAN valid"] = iban_ok(rec["IBAN"]) if rec["IBAN"] else None
    print({k: v for k, v in rec.items() if v is not None})
