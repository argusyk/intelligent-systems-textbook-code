def is_var(t):
    return isinstance(t, str) and t[:1].isupper()

def walk(t, s):
    while is_var(t) and t in s:
        t = s[t]
    return t

def occurs(v, t, s):
    t = walk(t, s)
    return t == v or (isinstance(t, tuple) and any(occurs(v, a, s) for a in t[1:]))

def unify(a, b, s=None):
    s = {} if s is None else s
    a, b = walk(a, s), walk(b, s)
    if a == b:
        return s
    if is_var(a):
        return None if occurs(a, b, s) else {**s, a: b}
    if is_var(b):
        return unify(b, a, s)
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
        for x, y in zip(a[1:], b[1:]):
            s = unify(x, y, s)
            if s is None:
                return None
        return s
    return None

print(unify(("egt", "X", 705), ("egt", "apu123", "T")))   # {'X': 'apu123', 'T': 705}
print(unify(("p", "X"), ("p", ("f", "X"))))               # None: перевірка входження
