import heapq

G = {"S": {"A": 4, "B": 2}, "A": {"C": 5, "D": 7}, "B": {"C": 8, "D": 10},
     "C": {"G": 6}, "D": {"G": 3}, "G": {}}
h = {"S": 11, "A": 9, "B": 10, "C": 6, "D": 3, "G": 0}

def astar(start, goal):
    open_ = [(h[start], 0, start, [start])]           # (f, g, vertex, path)
    best_g = {start: 0}
    while open_:
        f, g, v, path = heapq.heappop(open_)
        if v == goal:
            return g, path
        for w, cost in G[v].items():
            g2 = g + cost
            if g2 < best_g.get(w, float("inf")):     # a cheaper path has been found
                best_g[w] = g2
                heapq.heappush(open_, (g2 + h[w], g2, w, path + [w]))
    return None

print(astar("S", "G"))   # (14, ['S', 'A', 'D', 'G'])
