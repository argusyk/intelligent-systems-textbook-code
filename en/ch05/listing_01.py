from itertools import product

def entails(kb, goal, variables):
    """kb: a list of functions of a dictionary of values; goal: a function."""
    for values in product([False, True], repeat=len(variables)):
        env = dict(zip(variables, values))
        if all(f(env) for f in kb) and not goal(env):
            return False, env            # a counterexample has been found
    return True, None

imp = lambda a, b: (not a) or b
kb = [lambda v: imp(v["e"] and v["s"], v["n"]), lambda v: v["e"], lambda v: v["s"]]
print(entails(kb, lambda v: v["n"], ["e", "s", "n"]))   # (True, None)
print(entails(kb[:2], lambda v: v["n"], ["e", "s", "n"]))
# (False, {'e': True, 's': False, 'n': False}) – without the fact s the conclusion is not guaranteed
