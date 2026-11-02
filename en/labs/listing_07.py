import numpy as np

def retrieval_metrics(ranked_lists, gold, k=3):
    """ranked_lists: for each question, identifiers in order of decreasing relevance."""
    recall = np.mean([g in r[:k] for r, g in zip(ranked_lists, gold)])
    mrr = np.mean([1 / (r.index(g) + 1) if g in r else 0 for r, g in zip(ranked_lists, gold)])
    return round(float(recall), 3), round(float(mrr), 3)

ranked = [["49-30", "49-20", "49-10"], ["49-10", "49-50", "49-60"], ["49-80", "49-70",
                                                                     "49-20"],
          ["49-20", "49-30", "49-40"]]
gold = ["49-30", "49-50", "49-70", "49-40"]
print("recall@3, MRR:", retrieval_metrics(ranked, gold))
print("recall@1, MRR:", retrieval_metrics(ranked, gold, k=1))
