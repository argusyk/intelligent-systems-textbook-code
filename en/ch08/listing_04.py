# STRIPS-style planning: replacing an APU starter. A state is a set of facts, an action has
# preconditions, an add list and a delete list; the plan is found by breadth-first search.
from collections import deque

ACTIONS = {  # name: (preconditions, add, delete)
    "de-energize APU":       ({"power on"}, {"de-energized"}, {"power on"}),
    "open cowling":          (set(), {"cowling open"}, {"cowling closed"}),
    "disconnect wiring":     ({"de-energized", "cowling open", "wiring connected"},
                              {"wiring disconnected"}, {"wiring connected"}),
    "remove starter":        ({"wiring disconnected", "old starter"}, {"starter removed"},
                              {"old starter"}),
    "install starter":       ({"starter removed", "new starter in stock"}, {"new starter"},
                              {"starter removed", "new starter in stock"}),
    "connect wiring":        ({"new starter", "wiring disconnected"},
                              {"wiring connected"},
                              {"wiring disconnected"}),
    "close cowling":         ({"cowling open"}, {"cowling closed"}, {"cowling open"}),
    "apply power":           ({"de-energized", "cowling closed"}, {"power on"},
                              {"de-energized"}),
    "test run":              ({"power on", "new starter", "wiring connected",
                               "cowling closed"},
                              {"start checked"}, set()),
}
START = frozenset({"power on", "cowling closed", "wiring connected", "old starter",
                   "new starter in stock"})
GOAL = {"start checked", "cowling closed"}

def plan(start, goal):
    queue, seen, expanded = deque([(start, [])]), {start}, 0
    while queue:
        state, steps = queue.popleft(); expanded += 1
        if goal <= state:
            return steps, expanded
        for name, (pre, add, rem) in ACTIONS.items():
            if pre <= state:
                nxt = frozenset((state - rem) | add)
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, steps + [name]))
    return None, expanded

steps, n = plan(START, GOAL)
print(f"states expanded: {n}; plan steps: {len(steps)}")
for i, a in enumerate(steps, 1):
    print(f"{i}. {a}")
