import anthropic

# knowledge base fragments found by the retrieval stage (retrieval itself is shown in case 16.9 and Section 17.8)
FOUND = [
    "R5: overheating together with a slow start indicates poor "
    "fuel atomisation by the nozzles.",
    "R12: increased fuel flow at normal speed is characteristic "
    "of fouled nozzles.",
    "R13: after the nozzles are flushed, hypothesis H1 is unlikely for 50 operating hours.",
    "Procedure: on-wing nozzle flushing, takes about 4 h, "
    "see the maintenance manual, section 49-30.",
]
diagnosis = "H1 (fouled nozzles), CF = 0.85; rules R5 and R12 fired"
context = "\n".join(FOUND)

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=600,
    system=("You explain an expert system's conclusion to a maintenance technician. Use only "
            "the knowledge base fragments provided. Do not change the diagnosis and do not "
            "add facts of your own."),
    messages=[{"role": "user",
               "content": f"System conclusion: {diagnosis}\n\n"
                          f"Knowledge base fragments:\n{context}\n\n"
                          "Briefly explain why the system reached this conclusion "
                          "and what to check first."}],
)
print("".join(b.text for b in resp.content if b.type == "text"))
