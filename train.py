import json
import os
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import apk
os.makedirs("reports", exist_ok=True)
K = 12
def load(path):
    return pd.read_parquet(path).sort_values(["week", "customer_id"]).reset_index(drop=True)
train = load("data/train_feat.parquet")
es = load("data/es_feat.parquet")
val = load("data/val_feat.parquet")
actual = (pd.read_parquet("data/val_actual.parquet")
            .groupby("customer_id")["article_id"].apply(list))
DROP = ["customer_id", "article_id", "label", "week"]
FEATURES = [c for c in train.columns if c not in DROP]
CATS = [c for c in ["product_type_no", "product_group_name", "graphical_appearance_no",
                    "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id",
                    "department_no", "index_group_no", "section_no", "garment_group_no",
                    "club_member_status", "fashion_news_frequency"] if c in FEATURES]
print(f"{len(FEATURES)} features | train rows {len(train):,} | positives {int(train['label'].sum()):,}")
def group_sizes(df):
    """LightGBM needs: how many consecutive rows belong to each customer (per week)."""
    return df.groupby(["week", "customer_id"], sort=False).size().to_numpy()
dtrain = lgb.Dataset(train[FEATURES], label=train["label"], group=group_sizes(train),
                     categorical_feature=CATS, free_raw_data=False)
des = lgb.Dataset(es[FEATURES], label=es["label"], group=group_sizes(es),
                  categorical_feature=CATS, reference=dtrain, free_raw_data=False)
params = dict(
    objective="lambdarank",      
    metric="map",
    eval_at=[K],
    learning_rate=0.05,
    num_leaves=15,               
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    seed=42,
    verbose=-1,
)
model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[des], valid_names=["es_week"],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
print("Best iteration:", model.best_iteration)
model.save_model("data/model.txt")
def top_k(df, score, k=K):
    d = df[["customer_id", "article_id"]].copy()
    d["score"] = np.asarray(score)
    d = d.sort_values(["customer_id", "score"], ascending=[True, False])
    d["rank"] = d.groupby("customer_id").cumcount() + 1
    return d[d["rank"] <= k]

def map_at_k(top, actual):
    lists = top.groupby("customer_id")["article_id"].apply(list)
    return float(np.mean([apk(items, lists.get(c, []), K) for c, items in actual.items()]))


scores = {

    "Best-sellers (baseline)": val["item_pop_7d"].to_numpy(),

    "Repurchase + popular": np.where(val["is_history"] == 1,
                                     1e6 - val["ui_last_days"].fillna(1e5), val["item_pop_7d"]),

    "LightGBM LambdaRank": model.predict(val[FEATURES], num_iteration=model.best_iteration),
}

results, recs = {}, []
for name, s in scores.items():
    top = top_k(val, s)
    results[name] = map_at_k(top, actual)
    recs.append(top.assign(method=name))
    print(f"{name:26s} MAP@12 = {results[name]:.5f}")

recall = float(val["label"].sum() / len(pd.read_parquet("data/val_actual.parquet")))
summary = {"map12": results, "scored_customers": int(len(actual)), "candidate_recall": recall,
           "best_iteration": int(model.best_iteration), "n_features": len(FEATURES)}
json.dump(summary, open("reports/results.json", "w"), indent=2)
pd.concat(recs).to_parquet("data/val_recs.parquet")
print(f"Scored customers: {len(actual):,} | candidate recall: {recall:.4f}")


imp = (pd.DataFrame({"feature": FEATURES, "gain": model.feature_importance("gain")})
         .sort_values("gain", ascending=False))
imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
imp.to_csv("reports/feature_importance.csv", index=False)
print("\nTop 15 features (share of total gain):")
print(imp.head(15)[["feature", "gain_pct"]].round(1).to_string(index=False))

top15 = imp.head(15).iloc[::-1]
plt.figure(figsize=(8, 5.5))
plt.barh(top15["feature"], top15["gain_pct"], color="#2B5C99")
plt.xlabel("Share of total gain (%)")
plt.title("LightGBM feature importance (top 15)")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=160)
print("Saved: data/model.txt, data/val_recs.parquet, reports/results.json, reports/feature_importance.png")