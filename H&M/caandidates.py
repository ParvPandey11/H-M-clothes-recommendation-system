import pandas as pd

tx = pd.read_parquet("data/transactions_small.parquet")
tx["t_dat"] = pd.to_datetime(tx["t_dat"])

def make_candidates(tx, cutoff, n_hist=30, n_pop=30):
    cutoff = pd.Timestamp(cutoff)
    history = tx[tx["t_dat"] < cutoff]

    past = (history.sort_values("t_dat", ascending=False).drop_duplicates(["customer_id", "article_id"]).groupby("customer_id").head(n_hist)[["customer_id", "article_id"]])
    past["source"] = "history"

    week = history[history["t_dat"] >= cutoff - pd.Timedelta(days=7)]
    pop_items = week["article_id"].value_counts().head(n_pop).index
    customers = tx["customer_id"].unique()
    pop = pd.MultiIndex.from_product(
        [customers, pop_items], names=["customer_id", "article_id"]
    ).to_frame(index=False)
    pop["source"] = "popular"

    cands = pd.concat([past, pop], ignore_index=True)
    cands = cands.drop_duplicates(["customer_id", "article_id"], keep="first")
    return cands, past, list(pop_items)
cands, past, pop_items = make_candidates(tx, "2020-09-16")
print("Total candidate rows:", cands.shape)
print("Avg candidates per customer:", len(cands) / cands["customer_id"].nunique())
val = tx[tx["t_dat"] >= "2020-09-16"]
val_pairs = val[["customer_id", "article_id"]].drop_duplicates()

m = val_pairs.merge(cands[["customer_id", "article_id"]],
                    on=["customer_id", "article_id"], how="left", indicator=True)
print("Candidate recall:", (m["_merge"] == "both").mean())

def apk(actual_items, predicted, k=12):
    actual_set = set(actual_items)
    hits, score = 0, 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_set), k)

past_lists = past.groupby("customer_id")["article_id"].apply(list)
actual = val.groupby("customer_id")["article_id"].apply(list)

scores = []
for cust, items in actual.items():
    pred = past_lists.get(cust, [])[:12]
    for p in pop_items:
        if len(pred) >= 12:
            break
        if p not in pred:
            pred.append(p)
    scores.append(apk(items, pred))

print("Repurchase + popular MAP@12:", sum(scores) / len(scores))