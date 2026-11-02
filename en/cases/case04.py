# Knowledge graph of an educational programme: prerequisites, inheritance and queries
import networkx as nx

G = nx.DiGraph()
prereq = [("Mathematics", "Discrete Mathematics"), ("Discrete Mathematics",
                                                    "Intelligent Systems"),
          ("Computer Science", "Programming"), ("Programming", "Intelligent Systems"),
          ("Mathematics", "Probability Theory"), ("Probability Theory",
                                                  "Intelligent Systems"),
          ("Intelligent Systems", "Machine Learning in Engineering")]
G.add_edges_from(prereq, rel="prerequisite")
kind = {"Mathematics": "fundamental", "Computer Science": "fundamental",
        "Discrete Mathematics": "fundamental", "Probability Theory": "fundamental",
        "Programming": "professional", "Intelligent Systems": "elective",
        "Machine Learning in Engineering": "elective"}
nx.set_node_attributes(G, kind, "type")

target = "Intelligent Systems"
print("must be studied before the course:", sorted(nx.ancestors(G, target)))
print("the course gives access to:", sorted(nx.descendants(G, target)))
print("recommended order:", list(nx.topological_sort(G)))
print("cycles in the curriculum:", list(nx.simple_cycles(G)) or "none")
