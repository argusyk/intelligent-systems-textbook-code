# Analysis of a Kelly repertory grid: the expert's ratings of APU faults on constructs
import numpy as np

elements = ["nozzles", "fuel filter", "starter", "thermocouple", "bearing",
            "oil pump"]
constructs = [("overheat", "normal EGT"), ("prolonged start", "normal start"),
              ("high fuel flow", "normal fuel flow"), ("vibration", "no vibration"),
              ("oil pressure drop", "pressure normal"), ("develops gradually",
                                                        "occurs suddenly")]
# ratings 1..5: 1 – closer to the left pole of the construct, 5 – to the right
G = np.array([[1, 2, 5, 5, 3],    # overheat ... normal EGT
              [1, 1, 1, 5, 5],    # prolonged start ...
              [1, 4, 5, 5, 5],    # high fuel flow ...
              [5, 5, 5, 5, 1],    # vibration ...
              [5, 5, 5, 5, 2],    # oil pressure drop ...
              [1, 2, 3, 4, 2]])   # gradually ... suddenly
G = np.column_stack([G, [5, 5, 5, 3, 1, 1]])     # oil pump
D = np.abs(G[:, :, None] - G[:, None, :]).sum(0)  # distance between faults
iu = np.triu_indices(len(elements), 1)
pairs = sorted(zip(D[iu], iu[0], iu[1]))[:3]
print("hardest to distinguish (small distance):")
for d, i, j in pairs:
    diff = [constructs[k][0] for k in range(len(constructs)) if abs(G[k, i] - G[k, j]) >= 2]
    print(f"  {elements[i]} - {elements[j]}: distance {d}; "
          f"distinguished by: {diff or 'no construct'}")
R = np.corrcoef(G)
k1, k2 = np.unravel_index(np.argmax(np.abs(R - np.eye(len(R)))), R.shape)
print(f"most correlated constructs: '{constructs[k1][0]}' and '{constructs[k2][0]}', r = "
      f"{R[k1, k2]:.2f}")
