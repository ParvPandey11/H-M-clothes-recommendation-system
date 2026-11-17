import pandas as pd
from common import make_candidates

tx = pd.read_parquet("data/transactions_small.parquet")
tx["t_dat"] = pd.to_datetime(tx["t_dat"])

def add_labels(cands, tx, start, end):
    # Exam week mein kya kharida gaya
    week = tx[(tx["t_dat"] >= start) & (tx["t_dat"] < end)]
    bought = week[["customer_id", "article_id"]].drop_duplicates()
    bought["label"] = 1
    # Candidates ke saath jodo: match hua to 1, nahi to 0
    out = cands.merge(bought, on=["customer_id", "article_id"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    return out

# Training: 9 Sep se pehle ka data dekh kar, 9-15 Sep ke labels
train_c, _, _ = make_candidates(tx, "2020-09-09")
train = add_labels(train_c, tx, "2020-09-09", "2020-09-16")

# Validation: 16 Sep se pehle ka data dekh kar, 16-22 Sep ke labels
val_c, _, _ = make_candidates(tx, "2020-09-16")
val = add_labels(val_c, tx, "2020-09-16", "2020-09-23")

for name, df in [("TRAIN", train), ("VAL", val)]:
    print(name, df.shape)
    print("  Positives (label=1):", df["label"].sum())
    print("  Positive ratio:", df["label"].mean())

train.to_parquet("data/train_base.parquet")
val.to_parquet("data/val_base.parquet")