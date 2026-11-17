"""Pure-pandas helpers (no Streamlit) so the logic can be tested separately."""
import os
import pandas as pd

def load_data(folder="demo_data"):
    articles = pd.read_csv(f"{folder}/articles.csv", dtype={"article_id": str}).set_index("article_id")
    tx = pd.read_csv(f"{folder}/transactions.csv", dtype={"article_id": str}, parse_dates=["t_dat"])
    recs = pd.read_csv(f"{folder}/recs.csv", dtype={"article_id": str})
    return articles, tx, recs

def bestsellers(tx, k=12, days=7):
    recent = tx[tx.t_dat >= tx.t_dat.max() - pd.Timedelta(days=days)]
    return recent.article_id.value_counts().head(k).index.tolist()

def recs_for(customer_id, recs, tx, k=12):
    """Returns (list of (article_id, score or None), is_cold_start)."""
    r = recs[recs.customer_id == customer_id].sort_values("rank").head(k)
    if len(r):
        return list(zip(r.article_id, r.score)), False
    return [(a, None) for a in bestsellers(tx, k)], True

def history_for(customer_id, tx, n=8):
    return tx[tx.customer_id == customer_id].sort_values("t_dat", ascending=False).head(n)

def image_path(article_id, img_dir="images"):
    p = os.path.join(img_dir, article_id[:3], f"{article_id}.jpg")
    return p if os.path.exists(p) else None