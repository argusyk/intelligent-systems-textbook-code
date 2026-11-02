# APU-Expert agent with tool use via the Claude API.
# Without the ANTHROPIC_API_KEY variable an offline imitation of the model (ScriptedClient) is used;
# it reproduces the same message format, so the agent loop can be tested without a key.
import json, os
from types import SimpleNamespace as NS

TOOLS = [
    {"name": "get_parameters", "description": "Returns the latest measured APU parameters for a tail number.",
     "input_schema": {"type": "object", "properties": {"unit": {"type": "string"}}, "required": ["unit"]}},
    {"name": "run_diagnosis", "description": "Runs the expert system and returns hypotheses with certainty factors.",
     "input_schema": {"type": "object", "properties": {"params": {"type": "object"}}, "required": ["params"]}},
]

def get_parameters(unit):                     # in a real system: a query to the recorder database
    return {"EGT": 705, "start_time": 52, "fuel_flow": 155, "vibration": 3.1}

def run_diagnosis(params):                    # simplified core from Chapter 10
    cf = 0.0
    def comb(a, b): return a + b * (1 - a)
    if params["EGT"] > 680 and params["start_time"] > 45: cf = comb(cf, 0.7)
    if params["fuel_flow"] > 150: cf = comb(cf, 0.5)
    return {"H1_nozzles": round(cf, 2), "rules": ["R5", "R12"]}

DISPATCH = {"get_parameters": get_parameters, "run_diagnosis": run_diagnosis}

class ScriptedClient:                         # offline imitation: the same response format
    def __init__(self):
        self.messages = self; self.step = 0
    def create(self, **kw):
        self.step += 1
        if self.step == 1:
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="get_parameters",
                                                           input={"unit": "UR-PSA"})])
        if self.step == 2:
            params = json.loads(kw["messages"][-1]["content"][0]["content"])
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t2", name="run_diagnosis",
                                                           input={"params": params})])
        res = json.loads(kw["messages"][-1]["content"][0]["content"])
        return NS(stop_reason="end_turn", content=[NS(type="text",
                  text=f"The most likely cause: fouled nozzles (CF = {res['H1_nozzles']}, "
                       f"rules {', '.join(res['rules'])}). I recommend inspecting and flushing the nozzles.")])

def client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()
    return ScriptedClient()

def run_agent(question, max_steps=5):
    c, messages = client(), [{"role": "user", "content": question}]
    for _ in range(max_steps):
        r = c.messages.create(model="claude-opus-4-8", max_tokens=800, tools=TOOLS,
                              system="You assist a maintenance technician. "
                                     "Take the diagnosis only from the run_diagnosis tool.",
                              messages=messages)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text")
        messages.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type == "tool_use":
                try:
                    out, err = DISPATCH[b.name](**b.input), False   # the program executes the tool
                except Exception as e:                              # unknown tool or wrong arguments
                    out, err = {"error": str(e)}, True
                print(f"  tool {b.name}({b.input}) -> {out}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, ensure_ascii=False), "is_error": err})
        messages.append({"role": "user", "content": results})
    return "step limit exceeded"

if __name__ == "__main__":
    print(run_agent("What is wrong with the APU of aircraft UR-PSA? Complaint: overheating."))
