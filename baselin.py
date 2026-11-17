import pandas as pd
tx = pd.read_parquet("data/transactions_small.parquet")
tx["t_dat"] = pd.to_datetime(tx["t_dat"])
val_start = pd.Timestamp("2020-09-16")
last_week_start = pd.Timestamp("2020-09-09")
last_week = tx[(tx["t_dat"] >= last_week_start) & (tx["t_dat"] < val_start)]
top12 = last_week["article_id"].value_counts().head(12).index.tolist()
val = tx[tx["t_dat"] >= val_start]
actual = val.groupby("customer_id")["article_id"].apply(list)
print("Customers who bought in validation week:", len(actual))
def apk(actual_items, predicted, k=12):
    actual_set = set(actual_items)
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_set), k)
scores = [apk(items, top12) for items in actual]
print("Baseline MAP@12:", sum(scores) / len(scores))