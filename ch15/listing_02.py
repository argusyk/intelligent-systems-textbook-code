# Мінімакс і альфа-бета-відсікання на грі «хрестики-нулики»: однаковий результат,
# різна кількість розглянутих позицій
import math
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4,
                                                                                       6)]

def winner(b):
    for i, j, k in LINES:
        if b[i] != " " and b[i] == b[j] == b[k]: return b[i]
    return None

def minimax(b, player, stats):
    stats[0] += 1
    w = winner(b)
    if w: return 1 if w == "X" else -1
    if " " not in b: return 0
    vals = []
    for m in [i for i, c in enumerate(b) if c == " "]:
        b[m] = player
        vals.append(minimax(b, "O" if player == "X" else "X", stats))
        b[m] = " "
    return max(vals) if player == "X" else min(vals)

def alphabeta(b, player, alpha, beta, stats):
    stats[0] += 1
    w = winner(b)
    if w: return 1 if w == "X" else -1
    if " " not in b: return 0
    best = -math.inf if player == "X" else math.inf
    for m in [i for i, c in enumerate(b) if c == " "]:
        b[m] = player
        v = alphabeta(b, "O" if player == "X" else "X", alpha, beta, stats)
        b[m] = " "
        if player == "X":
            best = max(best, v); alpha = max(alpha, v)
        else:
            best = min(best, v); beta = min(beta, v)
        if alpha >= beta: break                    # відсікання: гілка не вплине на результат
    return best

for name, f in [("мінімакс", lambda b, s: minimax(b, "X", s)),
                ("альфа-бета", lambda b, s: alphabeta(b, "X", -math.inf, math.inf, s))]:
    stats = [0]
    v = f([" "] * 9, stats)
    print(f"{name:11s} ціна гри = {v:+d}, розглянуто позицій: {stats[0]}")
