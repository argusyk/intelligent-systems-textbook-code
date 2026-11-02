# Multi-agent "orchestrator – workers – critic" scheme for APU-Expert.
# Each agent has one role and minimal rights; the agents exchange structured messages.
# Offline, the worker agents are ordinary functions; in a real system the diagnostician and the
# librarian may be separate language model calls with their own tools (agent.py, rag.py).
import re
from agent import get_parameters, run_diagnosis
from rag import Index, build

INDEX = Index(build(2, 1))
LOG = []

def send(frm, to, kind, body):
    LOG.append((frm, to, kind, body))
    return body

def diagnostician(task):                       # rights: read parameters, run the expert system
    p = get_parameters(task["unit"])
    return {"params": p, **run_diagnosis(p)}

def librarian(task):                           # rights: manual search only
    return [(sec, text) for sec, text, _ in INDEX.search(task["query"], k=2)]

def critic(draft, evidence):                   # rights: read the draft and the evidence only
    problems = []
    if not re.search(r"\[\d\d-\d\d\]", draft): problems.append("no reference to the manual")
    if re.search(r"release\w*\s+(to service\s+)?(can|is allowed)", draft.lower()):
        problems.append("the agent cannot grant release to service")
    if "CF" not in draft: problems.append("no certainty factor")
    cited = set(re.findall(r"\[(\d\d-\d\d)\]", draft))
    if not cited <= {s for s, _ in evidence}: problems.append("reference to a section that was not retrieved")
    return problems

def orchestrator(unit, question, max_rounds=2):
    d = send("orchestrator", "diagnostician", "task", {"unit": unit})
    diag = send("diagnostician", "orchestrator", "result", diagnostician(d))
    q = send("orchestrator", "librarian", "task", {"query": "nozzle flushing procedure duration"})
    ev = send("librarian", "orchestrator", "result", librarian(q))
    draft = (f"Unit {unit}: probable cause is fouled nozzles, "
             f"CF = {diag['H1_nozzles']} (rules {', '.join(diag['rules'])}). "
             "Other systems are normal, release to service can be granted.")   # a deliberate error
    for rnd in range(max_rounds):
        issues = send("critic", "orchestrator", "remarks", critic(draft, ev))
        if not issues:
            return draft
        draft = (f"Unit {unit}: probable cause is fouled nozzles, "
                 f"CF = {diag['H1_nozzles']} (rules {', '.join(diag['rules'])}). "
                 f"On-wing flushing is recommended, about 4 h [{ev[0][0]}]. "
                 "The decision on release to service is made by a certified technician [49-90].")
        ev = ev + [(s, t) for s, t, _ in INDEX.search("release to service decision certified technician", k=1)]
    return draft + " (the critic has remarks: human review required)"

if __name__ == "__main__":
    answer = orchestrator("UR-PSA", "Why is the start slow and what should be done?")
    for frm, to, kind, body in LOG:
        text = str(body)
        print(f"{frm:>13s} -> {to:<13s} {kind:8s} {text[:70]}{'...' if len(text) > 70 else ''}")
    print("\nResult:", answer)
