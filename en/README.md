# Companion code for the textbook "Intelligent Systems"

```
pip install -r requirements.txt
python cases/case07.py
```

- `chNN/listing_KK.*` – listings from the text of Chapter NN in order of appearance (`.py` – Python, `.clp` – CLIPS, `.pl` – Prolog, `.ttl` – Turtle).
- `labs/` – fragments from the worked examples of the laboratory works (listing_04 and listing_05 complement the code of Chapters 12 and 13 and are not run separately).
- `cases/caseNN.py` – the case studies of the chapters (the "Case Study" sections).
- `_report.txt` – the result of an automatic run of all listings (`py build/extract_code.py`).

Check the Prolog listings in SWI-Prolog (https://www.swi-prolog.org). The examples with the Claude API
run only if the environment variable `ANTHROPIC_API_KEY` is set.

- `chNN/*.py` for Chapters 17–26 – complete programs whose fragments are given in the text.
- `ch19/ukr_text.py` deliberately works with Ukrainian text: it demonstrates the preprocessing an inflected language requires (Section 19.4).
- `data/` – loaders for the open datasets NASA C-MAPSS (`cmapss.py`) and MovieLens 100K (`movielens.py`);
  download the datasets themselves manually from the NASA and GroupLens websites; without them the programs work on demonstration files.
- `tests/tests_moodle.gift` – self-assessment tests in GIFT format for import into the Moodle question bank.
