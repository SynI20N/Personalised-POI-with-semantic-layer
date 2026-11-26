"""
Quality check of the saved k-means clusters (read-only; prints numbers for the report).

    python src/cluster_quality.py

1. Size: categories per cluster, and NYC training POIs per cluster (the geography fallback and the
   semantic TLDA need enough POIs per cluster).
2. Taxonomy: how many clusters stay inside one root category, and which clusters mix roots.
3. Cohesion: mean cosine between categories in the same cluster vs in different clusters.
"""
import numpy as np
import pandas as pd

category_vectors = pd.read_parquet("data/processed/category_embeddings.parquet")
category_clusters = pd.read_parquet("data/processed/category_clusters_K50.parquet")
nyc_poi_map = pd.read_csv("data/processed/nyc/poi_map.csv")

categories = category_vectors.merge(category_clusters[["category_id", "cluster"]], on="category_id")
print(f"categories: {len(category_vectors)} embedded, {len(categories)} with a cluster")
categories["root"] = categories["full_category_path"].str.split(" > ").str[0]

# 1. size
categories_per_cluster = categories.groupby("cluster").size()
nyc_train_pois = nyc_poi_map[nyc_poi_map["in_train"] == 1]
nyc_train_pois = nyc_train_pois.merge(categories[["category_id", "cluster"]],
                                      left_on="categoryId", right_on="category_id", how="left")
print(f"NYC training POIs: {len(nyc_train_pois)}, without a cluster: {nyc_train_pois['cluster'].isna().sum()}")
train_pois_per_cluster = nyc_train_pois.groupby("cluster").size().reindex(categories_per_cluster.index, fill_value=0)
print(f"categories per cluster: min {categories_per_cluster.min()}, median {categories_per_cluster.median():.0f}, "
      f"max {categories_per_cluster.max()}")
print(f"NYC training POIs per cluster: median {train_pois_per_cluster.median():.0f}, "
      f"max {train_pois_per_cluster.max()} ({train_pois_per_cluster.max() / len(nyc_train_pois):.0%} of all), "
      f"< 20: {(train_pois_per_cluster < 20).sum()}, none: {(train_pois_per_cluster == 0).sum()}")

# 2. taxonomy: share of the cluster's categories that belong to its most common root
root_purity = categories.groupby("cluster")["root"].agg(lambda roots: roots.value_counts(normalize=True).iloc[0])
print(f"clusters inside one root: {(root_purity == 1).sum()} / {len(root_purity)}, "
      f"mixed (< 70 % one root): {(root_purity < 0.7).sum()}, mean root purity {root_purity.mean():.2f}")
for cluster_id in root_purity[root_purity < 0.7].index:
    members = categories[categories["cluster"] == cluster_id]
    print(f"  mixed cluster {cluster_id} ({', '.join(members['root'].unique())}): "
          + "; ".join(members["category_name"].head(10)))

# 3. cohesion: vectors are unit length, so the dot product is the cosine
vectors = np.vstack(categories["embedding"].to_numpy())
cosine = vectors @ vectors.T
same_cluster = categories["cluster"].to_numpy()[:, None] == categories["cluster"].to_numpy()[None, :]
not_self = ~np.eye(len(categories), dtype=bool)
print(f"mean cosine, same cluster: {cosine[same_cluster & not_self].mean():.3f}, "
      f"different clusters: {cosine[~same_cluster].mean():.3f}")
