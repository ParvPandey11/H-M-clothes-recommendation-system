"""Creates small FAKE data in demo_data/ so you can test app.py before using real data."""
import json, os
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
os.makedirs("demo_data", exist_ok=True)

types = ["Trousers", "Dress", "Sweater", "T-shirt", "Jacket", "Skirt"]
colors = ["Black", "White", "Blue", "Red", "Beige", "Green"]
depts = ["Ladieswear", "Menswear", "Divided", "Kids"]

n_art, n_cust = 500, 200
articles = pd.DataFrame({
    "article_id": [f"{108775015 + i:010d}" for i in range(n_art)],
    "prod_name": [f"Item {i}" for i in range(n_art)],
    "product_type_name": rng.choice(types, n_art),
    "colour_group_name": rng.choice(colors, n_art),
    "department_name": rng.choice(depts, n_art),
})
articles.to_csv("demo_data/articles.csv", index=False)

cust = [f"cust_{i:04d}" for i in range(n_cust)]
tx = pd.DataFrame({
    "customer_id": rng.choice(cust, 3000),
    "article_id": rng.choice(articles.article_id, 3000),
    "t_dat": pd.to_datetime("2020-09-01") + pd.to_timedelta(rng.integers(0, 22, 3000), unit="D"),
    "price": rng.uniform(0.01, 0.12, 3000).round(4),
})
tx.to_csv("demo_data/transactions.csv", index=False)

# recs from the "model": 12 per customer, descending score. Leave 10 customers out (cold start)
rows = []
for c in cust[:-10]:
    items = rng.choice(articles.article_id, 12, replace=False)
    scores = np.sort(rng.random(12))[::-1]
    rows += [(c, a, float(s), r + 1) for r, (a, s) in enumerate(zip(items, scores))]
pd.DataFrame(rows, columns=["customer_id", "article_id", "score", "rank"]).to_csv("demo_data/recs.csv", index=False)

json.dump({"scores": {"Best-sellers (baseline)": 0.0078, "Repurchase + popular": 0.0240,
                      "LightGBM LambdaRank": 0.0263}}, open("demo_data/results.json", "w"))
print("fake data written")