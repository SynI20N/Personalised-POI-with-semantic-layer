# There are 1279 category embeddings, each represents a foursqaure category path. Since its a cetegory tree
# We cna shoose k based on the number of nodes in layer n, st, n controls the
# granularity of the clusters 
from collections import Counter
import numpy as np

import pandas as pd
from collections import defaultdict

PARQUET_PATH = "data/raw/categories_000000.parquet"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Optional: Prevent long strings inside individual cells from being cut off
pd.set_option('display.max_colwidth', None)
df = pd.read_parquet(r"data\raw\categories_000000.parquet")
print(df.head())



import pandas as pd
from collections import defaultdict

# PARQUET_PATH = "data/category_embeddings.parquet"

df = pd.read_parquet(PARQUET_PATH)

# Store the hierarchy
nodes = {}
children = defaultdict(set)

for _, row in df.iterrows():
    parent_id = ""

    for level in range(1, 7):
        category_id = row.get(f"level{level}_category_id")
        category_name = row.get(f"level{level}_category_name")

        if pd.isna(category_id) or pd.isna(category_name):
            break

        category_id = str(category_id)

        nodes[category_id] = {
            "id": category_id,
            "label": str(category_name),
            "depth": level,
            "parent": parent_id,
        }

        children[parent_id].add(category_id)
        parent_id = category_id

# Find terminal categories
leaves = {
    node_id for node_id in nodes
    if not children[node_id]
}

# Count descendant leaves for each node
def count_leaves(node_id):
    if node_id in leaves:
        return 1

    return sum(
        count_leaves(child)
        for child in children[node_id]
    )

for node_id, node in nodes.items():
    node["leaf_count"] = count_leaves(node_id)

# Count nodes per layer
layer_counts = (
    pd.DataFrame(nodes.values())
    .groupby("depth")
    .size()
)

print("Nodes per layer:")
print(layer_counts)

print(f"\nTotal categories: {len(nodes)}")
print(f"Terminal categories: {len(leaves)}")

for depth in range(1, 6):
    parents = defaultdict(set)

    for node_id, node in nodes.items():
        if node["depth"] == depth + 1:
            parents[node["parent"]].add(node_id)

    counts = [len(v) for v in parents.values()]

    if counts:
        print(f"\nDepth {depth} -> {depth + 1}")
        print(f"Parents: {len(counts)}")
        print(f"Mean children: {np.mean(counts):.1f}")
        print(f"Median children: {np.median(counts):.1f}")
        print(f"Max children: {max(counts)}")
import plotly.graph_objects as go

plot_data = pd.DataFrame(nodes.values())

fig = go.Figure(go.Icicle(
    ids=plot_data["id"],
    labels=plot_data["label"],
    parents=plot_data["parent"],
    values=plot_data["leaf_count"],
    branchvalues="total",
    customdata=plot_data[["depth", "leaf_count"]],
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Depth: %{customdata[0]}<br>"
        "Descendant leaves: %{customdata[1]}"
        "<extra></extra>"
    ),
    tiling=dict(orientation="v"),
    maxdepth=3,
))

fig.update_layout(
    title="Foursquare Category Taxonomy",
    margin=dict(t=60, l=10, r=10, b=10),
    height=850,
)

fig.show()

# Save as standalone interactive HTML
fig.write_html("foursquare_taxonomy.html")


# def count_nodes_per_layer(category_paths, separator=" > "):
#     counts = Counter()

#     for path in category_paths:
#         parts = path.split(separator)

#         for depth in range(1, len(parts) + 1):
#             prefix = tuple(parts[:depth])
#             counts[depth] += 1 if prefix not in seen[depth] else 0
#             seen[depth].add(prefix)

#     return dict(counts)

"""
The Foursquare taxonomy analysis reveals a highly uneven hierarchical structure 
across its 1,279 categories, with 11 root categories expanding into 457 second-level nodes, 
followed by 474 third-level, 241 fourth-level, 82 fifth-level and 14 sixth-level nodes. 
The largest expansion occurs between the first and second levels, with an average of 41.5 
children per parent (median: 32), while subsequent levels generally have smaller branching factors,
although some branches remain considerably more complex than others. This suggests that clustering 
into just 11 groups would be too coarse, whereas using the full second-level granularity (457 groups)
would likely be excessive. Based on these observations, we propose testing K-means with 
K = 30, 40, 50, 60, 70, 80, 90, 100, 120 and 150, with particular emphasis on the range of 60–80 clusters as an initial hypothesis. 
These values will be evaluated using clustering quality metrics, cluster size distributions and semantic coherence to identify a 
suitable balance between granularity and interpretability.
"""

"""
Hyperparameters: 
- n-clusters: 30-150 test candidates
_clusters	30–150	Test the candidate values from your taxonomy analysis
init	k-means++	Better initial centroid placement
n_init	20–30	Reduces sensitivity to random initialisation
max_iter	300	Standard, generally sufficient
tol	1e-4	Standard convergence tolerance
random_state	42	Reproducibility

"""


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def cluster_embeddings(
    embeddings: pd.DataFrame,
    k: int,
    n_init: int = 20,
    max_iter: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster category embeddings using K-means.

    Args:
        embeddings: DataFrame containing an 'embedding' column.
        k: Number of clusters.
        n_init: Number of initialisations.
        max_iter: Maximum number of iterations.
        random_state: Random seed.

    Returns:
        Original DataFrame with an additional 'cluster' column.
    """

    # Convert embeddings into a numerical matrix
    X = np.vstack(embeddings["embedding"].values)

    # Normalise embeddings (L2)
    X = normalize(X, norm="l2")

    # Initialise K-means
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_state,
        algorithm="lloyd",
    )

    # Fit and predict clusters
    labels = kmeans.fit_predict(X)

    # Add cluster assignments
    result = embeddings.copy()
    result["cluster"] = labels

    return result


