import pandas as pd
from collections import defaultdict
import itertools

def load_co_purchase_table(orders_csv='orders.csv'):
    orders = pd.read_csv(orders_csv)
    co_purchase = defaultdict(lambda: defaultdict(int))

    for _, row in orders.iterrows():
        # Parse "1,2,15" into [1, 2, 15]
        product_ids = [int(pid.strip()) for pid in str(row['product_ids']).split(',')]

        # For every pair of products bought together, increase their co-purchase count
        for a, b in itertools.permutations(product_ids, 2):
            co_purchase[a][b] += 1

    return co_purchase

def get_also_bought(product_id, co_purchase, top_k=5):
    """
    Given a product ID, return the products most frequently
    bought together with it, ranked by frequency.
    """
    if product_id not in co_purchase:
        return []

    related = co_purchase[product_id]
    sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
    return sorted_related[:top_k]  # list of (product_id, count)

# Quick test
if __name__ == "__main__":
    co_purchase = load_co_purchase_table()

    products = pd.read_csv('products_processed.csv')

    test_product_id = 1  # change this to test different products
    results = get_also_bought(test_product_id, co_purchase)

    print(f"Products frequently bought with product ID {test_product_id}:\n")
    for pid, count in results:
        name = products[products['id'] == pid]['name'].values
        name = name[0] if len(name) > 0 else "Unknown"
        print(f"  Product {pid} ({name}) — bought together {count} times")