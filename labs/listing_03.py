import math
from collections import Counter

def entropy(rows):
    n = len(rows)
    return -sum(c / n * math.log2(c / n) for c in Counter(r[-1] for r in rows).values())

def gain(rows, a):
    groups = {}
    for r in rows:
        groups.setdefault(r[a], []).append(r)
    rem = sum(len(g) / len(rows) * entropy(g) for g in groups.values())
    return entropy(rows) - rem, groups

def id3(rows, attrs, names):
    labels = Counter(r[-1] for r in rows)
    if len(labels) == 1 or not attrs:
        return labels.most_common(1)[0][0]            # лист
    best = max(attrs, key=lambda a: gain(rows, a)[0])
    _, groups = gain(rows, best)
    rest = [a for a in attrs if a != best]
    return {names[best]: {v: id3(g, rest, names) for v, g in groups.items()}}
