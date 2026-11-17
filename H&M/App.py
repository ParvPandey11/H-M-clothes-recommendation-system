# import json, os
# import streamlit as st
# from demo_utils import load_data, recs_for, history_for, image_path

# DATA_DIR = os.environ.get("DEMO_DATA", "demo_data")
# IMG_DIR = os.environ.get("DEMO_IMAGES", "images")

# st.set_page_config(page_title="Fashion Recommender", layout="wide")

# @st.cache_data
# def get_data():
#     return load_data(DATA_DIR)

# articles, tx, recs = get_data()

# def show_item(col, article_id, score=None):
#     with col:
#         p = image_path(article_id, IMG_DIR)
#         if p:
#             st.image(p, use_container_width=True)
#         else:
#             st.markdown("🖼️ *(no image)*")
#         if article_id in articles.index:
#             a = articles.loc[article_id]
#             st.caption(f"**{a.prod_name}**  \n{a.product_type_name} · {a.colour_group_name}")
#         else:
#             st.caption(article_id)
#         if score is not None:
#             st.caption(f"score: {score:.3f}")

# def grid(items, ncols=4):
#     for i in range(0, len(items), ncols):
#         cols = st.columns(ncols)
#         for col, (aid, sc) in zip(cols, items[i:i + ncols]):
#             show_item(col, aid, sc)

# st.title("👗 Fashion Recommender (LightGBM LambdaRank)")

# all_customers = sorted(set(tx.customer_id) | set(recs.customer_id))
# with st.sidebar:
#     st.header("Pick a customer")
#     cid = st.selectbox("Customer", all_customers)
#     k = st.slider("How many recommendations", 4, 12, 12, step=4)
#     rpath = os.path.join(DATA_DIR, "results.json")
#     if os.path.exists(rpath):
#         st.header("Model comparison (MAP@12)")
#         scores = json.load(open(rpath)).get("scores", {})
#         st.bar_chart(scores)

# st.subheader("Recently bought")
# hist = history_for(cid, tx)
# if len(hist):
#     grid([(a, None) for a in hist.article_id], ncols=4)
# else:
#     st.info("No purchase history for this customer.")

# st.subheader("Recommended for you")
# items, cold = recs_for(cid, recs, tx, k)
# if cold:
#     st.warning("Cold start: no model recommendations, showing this week's best-sellers.")
# grid(items, ncols=4)
import pandas as pd
for f in ["val_recs", "customer_codes", "articles", "transactions_small"]:
    df = pd.read_parquet(f"data/{f}.parquet")
    print("\n==", f, df.shape)
    print(list(df.columns))
    print(df.head(2).to_string())
    