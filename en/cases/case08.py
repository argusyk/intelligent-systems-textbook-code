# A* on a grid: a technician's route from the hangar to the stand avoiding restricted zones
import heapq

GRID = ["S....#....",
        ".##..#.##.",
        ".#...#..#.",
        ".#.###..#.",
        "...#...##.",
        ".#.#.#....",
        ".#...#.##G"]
R, C = len(GRID), len(GRID[0])
find = lambda ch: next((r, c) for r in range(R) for c in range(C) if GRID[r][c] == ch)
start, goal = find("S"), find("G")
h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])      # Manhattan, admissible

def astar():
    open_, g, parent, expanded = [(h(start), 0, start)], {start: 0}, {start: None}, 0
    while open_:
        f, gc, p = heapq.heappop(open_)
        if p == goal:
            path = []
            while p: path.append(p); p = parent[p]
            return path[::-1], expanded
        expanded += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dr, p[1] + dc)
            if (0 <= q[0] < R and 0 <= q[1] < C and GRID[q[0]][q[1]] != "#"
                    and gc + 1 < g.get(q, 1e9)):
                g[q], parent[q] = gc + 1, p
                heapq.heappush(open_, (gc + 1 + h(q), gc + 1, q))
    return None, expanded

path, n = astar()
print(f"route length {len(path) - 1} steps, vertices expanded {n}")
for r in range(R):
    print("".join("*" if (r, c) in path[1:-1] else GRID[r][c] for c in range(C)))
