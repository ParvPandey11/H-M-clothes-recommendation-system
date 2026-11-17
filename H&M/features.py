
import numpy as np
import pandas as pd
from common import make_candidates, add_labels

TRAIN_CUTOFFS = ["2020-08-19", "2020-08-26", "2020-09-02"]
ES_CUTOFF = "2020-09-09"
VAL_CUTOFF = "2020-09-16"
NEG_PER_GROUP = 20      
SEED = 42

tx = pd.read_parquet("data/transactions_small.parquet")
tx["t_dat"] = pd.to_datetime(tx["t_dat"])
articles = pd.read_parquet("data/articles.parquet")
customers = pd.read_parquet("data/customers.parquet")

codes = pd.Index(tx["customer_id"].unique())
pd.DataFrame({"code": np.arange(len(codes), dtype="int32"),
              "customer_id": codes}).to_parquet("data/customer_codes.parquet")
tx["customer_id"] = codes.get_indexer(tx["customer_id"]).astype("int32")

ART_COLS = ["product_type_no", "product_group_name", "graphical_appearance_no",
            "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id",
            "department_no", "index_group_no", "section_no", "garment_group_no"]
art = articles[["article_id"] + ART_COLS].copy()
art["product_group_name"] = art["product_group_name"].astype("category").cat.codes
for c in ART_COLS:
    art[c] = art[c].astype("int32")

cust = customers.copy()
cust["customer_id"] = codes.get_indexer(cust["customer_id"])
cust = cust[cust["customer_id"] >= 0].copy()
cust["customer_id"] = cust["customer_id"].astype("int32")
for c in ["club_member_status", "fashion_news_frequency"]:
    cust[c] = cust[c].astype("category").cat.codes.astype("int32")   # -1 = missing
cust["FN"] = cust["FN"].fillna(0).astype("int32")
cust["Active"] = cust["Active"].fillna(0).astype("int32")
cust = cust[["customer_id", "age", "club_member_status", "fashion_news_frequency", "FN", "Active"]]


def add_features(df, cutoff):
    """Add item, user and user-item features to the candidate table using data BEFORE cutoff only."""
    cutoff = pd.Timestamp(cutoff)
    hist = tx[tx["t_dat"] < cutoff].merge(art, on="article_id", how="left")
    hist["age_days"] = (cutoff - hist["t_dat"]).dt.days

    # ---- item features
    g = hist.groupby("article_id")
    item = pd.DataFrame({
        "item_pop_all": g.size(),
        "item_pop_7d": hist[hist["age_days"] <= 7].groupby("article_id").size(),
        "item_pop_28d": hist[hist["age_days"] <= 28].groupby("article_id").size(),
        "item_avg_price": g["price"].mean(),
        "item_last_sold_days": g["age_days"].min(),
    }).reset_index()
    for c in ["item_pop_all", "item_pop_7d", "item_pop_28d"]:
        item[c] = item[c].fillna(0)

    
    g = hist.groupby("customer_id")
    user = pd.DataFrame({
        "user_n": g.size(),
        "user_n_items": g["article_id"].nunique(),
        "user_avg_price": g["price"].mean(),
        "user_last_days": g["age_days"].min(),
        "user_first_days": g["age_days"].max(),
        "user_n_28d": hist[hist["age_days"] <= 28].groupby("customer_id").size(),
    }).reset_index()
    user["user_n_28d"] = user["user_n_28d"].fillna(0)

    ui = (hist.groupby(["customer_id", "article_id"])
              .agg(ui_n=("price", "size"), ui_last_days=("age_days", "min")).reset_index())

    out = df.merge(art, on="article_id", how="left")
    out = out.merge(item, on="article_id", how="left")
    out = out.merge(user, on="customer_id", how="left")
    out = out.merge(ui, on=["customer_id", "article_id"], how="left")
    out = out.merge(cust, on="customer_id", how="left")

    for key, name in [("product_type_no", "type"), ("colour_group_code", "colour"), ("department_no", "dept")]:
        t = (hist.groupby(["customer_id", key])
                 .agg(**{f"{name}_n": ("price", "size"), f"{name}_last_days": ("age_days", "min")})
                 .reset_index())
        out = out.merge(t, on=["customer_id", key], how="left")

    for c in ["user_n", "user_n_items", "ui_n", "type_n", "colour_n", "dept_n"]:
        out[c] = out[c].fillna(0)

    out["type_share"] = out["type_n"] / out["user_n"].replace(0, np.nan)
    out["price_ratio"] = out["item_avg_price"] / out["user_avg_price"]
    out["is_history"] = (out["source"] == "history").astype("int32")
    out["ui_last_rank"] = out.groupby("customer_id")["ui_last_days"].rank(method="min")

    out = out.drop(columns=["source"])
    num = out.select_dtypes("float64").columns
    out[num] = out[num].astype("float32")
    return out.sort_values("customer_id").reset_index(drop=True)


def build_week(cutoff, mode):
    """mode = 'train' (smaller, for learning) or 'eval' (full, for scoring)."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=7)
    cands, _, _ = make_candidates(tx, cutoff)
    df = add_labels(cands, tx, cutoff, end)
    cands_cols = ["customer_id", "article_id", "label", "source"]
    df = df[cands_cols]

    if mode == "train":
        pos_customers = df.loc[df["label"] == 1, "customer_id"].unique()
        df = df[df["customer_id"].isin(pos_customers)]
        rng = np.random.default_rng(SEED)
        neg = df[df["label"] == 0].copy()
        neg["r"] = rng.random(len(neg))
        neg = neg[neg.groupby("customer_id")["r"].rank(method="first") <= NEG_PER_GROUP].drop(columns="r")
        df = pd.concat([df[df["label"] == 1], neg], ignore_index=True)
    else:
        buyers = tx.loc[(tx["t_dat"] >= cutoff) & (tx["t_dat"] < end), "customer_id"].unique()
        df = df[df["customer_id"].isin(buyers)]

    df = add_features(df, cutoff)
    print(f"  cutoff {cutoff.date()}: rows={len(df):,}  positives={int(df['label'].sum()):,}  "
          f"customers={df['customer_id'].nunique():,}")
    return df


if __name__ == "__main__":
    print("Building TRAIN weeks...")
    train = pd.concat([build_week(c, "train").assign(week=i) for i, c in enumerate(TRAIN_CUTOFFS)],
                      ignore_index=True)
    train.to_parquet("data/train_feat.parquet")
    print("TRAIN saved:", train.shape)

    print("Building ES week (early stopping)...")
    es = build_week(ES_CUTOFF, "train").assign(week=0)
    es.to_parquet("data/es_feat.parquet")

    print("Building VAL week (final exam)...")
    val = build_week(VAL_CUTOFF, "eval").assign(week=0)
    val.to_parquet("data/val_feat.parquet")

    v0 = pd.Timestamp(VAL_CUTOFF)
    actual = tx[(tx["t_dat"] >= v0) & (tx["t_dat"] < v0 + pd.Timedelta(days=7))]
    actual[["customer_id", "article_id"]].drop_duplicates().to_parquet("data/val_actual.parquet")
    print("VAL saved:", val.shape)
    print("Feature columns:", [c for c in val.columns if c not in ("customer_id", "article_id", "label", "week")])