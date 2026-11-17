import pandas as pd
def make_candidates(tx, cutoff, n_hist=30, n_pop=30):
    """Shortlist items for every customer using ONLY data before `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    history = tx[tx["t_dat"] < cutoff]

    past = (history.sort_values("t_dat", ascending=False)
                   .drop_duplicates(["customer_id", "article_id"])
                   .groupby("customer_id").head(n_hist)
                   [["customer_id", "article_id"]].copy())
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


def add_labels(cands, tx, start, end):
    """label = 1 if the customer bought the item between start (incl.) and end (excl.)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    week = tx[(tx["t_dat"] >= start) & (tx["t_dat"] < end)]
    bought = week[["customer_id", "article_id"]].drop_duplicates()
    bought["label"] = 1
    out = cands.merge(bought, on=["customer_id", "article_id"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    return out


def apk(actual_items, predicted, k=12):
    """Average precision at k for ONE customer."""
    actual_set = set(actual_items)
    hits, score = 0, 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_set), k)