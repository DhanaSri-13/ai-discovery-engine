import pandas as pd
from collections import defaultdict
import itertools

def load_co_purchase_table(orders_csv='orders.csv'):
    orders = pd.read_csv(orders_csv)
    co_purchase = defaultdict(lambda: defaultdict(int))

    for _, row in orders.iterrows():
        product_ids = [int(pid.strip()) for pid in str(row['product_ids']).split(',')]
        for a, b in itertools.permutations(product_ids, 2):
            co_purchase[a][b] += 1

    # Convert to plain dicts so it's picklable (needed for Streamlit caching)
    plain_dict = {k: dict(v) for k, v in co_purchase.items()}
    return plain_dict

def get_also_bought(product_id, co_purchase, top_k=5):
    if product_id not in co_purchase:
        return []

    related = co_purchase[product_id]
    sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
    return sorted_related[:top_k]