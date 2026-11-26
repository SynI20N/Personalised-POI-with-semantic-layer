"""
Embed every Foursquare category path once with a frozen sentence encoder.

All categories of the taxonomy are embedded (not only those in one city), so the same
file serves NYC and Tokyo. The text of a category is its full breadcrumb, e.g.
"Dining and Drinking > Bar > Sake Bar" (the category name if the path is missing).

Output: ``data/processed/category_embeddings.parquet`` with columns
category_id, category_name, full_category_path, embedding (list of floats).

    python src/path_to_embedding.py --model all-MiniLM-L6-v2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from src.construct_category_path import PROCESSED_DIR, load_category_taxonomy  # noqa: E402

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_FILE = PROCESSED_DIR / "category_embeddings.parquet"


def category_texts(taxonomy):
    """One row per category with the text to embed: the full path, else the name."""
    texts = taxonomy[["category_id", "category_name", "full_category_path"]].copy()
    texts = texts.drop_duplicates("category_id").reset_index(drop=True)
    texts["text"] = texts["full_category_path"].fillna(texts["category_name"])
    return texts


def embed_texts(texts, model_name=DEFAULT_MODEL, normalize=True):
    """Encode texts with a frozen sentence-transformers model; returns (n, dim) array."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), normalize_embeddings=normalize, show_progress_bar=True)
    
        

def embed_texts_weighted(paths, model_name=DEFAULT_MODEL, normalize=True, leaf_boost=1.0):
    """Embed category paths with a frozen sentence-transformers model; returns (n, dim) array.

    Each path is a sequence of level names from general to specific, e.g.
    ("Dining and Drinking", "Bar", "Sake Bar"). Every level is encoded and the levels are
    combined with a weighted mean (Eq. 5) instead of an equal-weight average.

    Leaf + parent only (D17): 0.7·leaf + 0.3·parent, L2-normalised; a root embeds its own name. 
    Embedding the full path would feed the taxonomy into the vector (see STATUS §6.4)."
    """
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    paths = [tuple(p.split(" > ")) for p in paths] 
    model = SentenceTransformer(model_name)

    # Encode each distinct level name once ("Bar" appears in many paths).
    names = sorted({name for p in paths for name in p})
    vecs = model.encode(names, normalize_embeddings=normalize, show_progress_bar=True)
    index = {name: i for i, name in enumerate(names)}
    

    out = []
    for p in paths:
        # fixed 0.7 - 0.3 weights 
        # w = np.arange(1, len(p) + 1, dtype=np.float64)  # linear weights 1..L
        # w[-1] *= leaf_boost                               # boost only the leaf
        # w /= w.sum()                                      # weighted mean ensure sum to one
        levels = p[-2:]                                         # leaf + parent only (D17)
        weights = np.array([0.3, 0.7]) if len(levels) == 2 else np.array([1.0])  # parent 0.3, leaf 0.7; a root is its own name
        category_vector = weights @ vecs[[index[name] for name in levels]]  # only leaf + parent
        out.append(category_vector / (np.linalg.norm(category_vector) + 1e-12))  # unit length, whatever the path depth

    return np.stack(out).astype(np.float32)

def build_category_embeddings(taxonomy, model_name=DEFAULT_MODEL, weighted=True):
    """Embedding table: category_id, category_name, full_category_path, embedding."""
    table = category_texts(taxonomy)
    if weighted==True: 
        table["embedding"] = embed_texts_weighted(table["text"], model_name).tolist()
    else: 

        table["embedding"] = embed_texts(table["text"], model_name).tolist()

    return table.drop(columns="text")


def load_category_embeddings(path=EMBEDDING_FILE):
    """
    Read the saved embeddings.

    Returns (table, matrix): the table without the embedding column, and the embeddings
    as a float array whose row i belongs to table row i.
    """
    import pandas as pd

    table = pd.read_parquet(path)
    matrix = np.vstack(table["embedding"].to_numpy()).astype(np.float32)
    return table.drop(columns="embedding"), matrix


# k-means over the category embeddings (moved from k_means.py; the taxonomy analysis stays there).
# One point per CATEGORY, not per POI, so frequent categories (e.g. "Home") do not pull the centroids.
# K: the taxonomy has 11 roots and 457 level-2 nodes, so 11 clusters is too coarse and ~450 just
# repeats the categories; candidates lie in between (NYC uses 398 categories).
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

    X = np.vstack(embeddings["embedding"].values)

    # A weighted mean of unit vectors is shorter than 1, by an amount that depends on the path.
    # L2-normalising makes k-means group by direction (cosine similarity), not by length.
    X = normalize(X, norm="l2")

    # k-means++ seeding plus 20 restarts: with ~1,300 points this is cheap and removes most of the
    # dependence on the random start; random_state makes the clusters reproducible.
    # runs k means 20 times and keep one with lowest intertia
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_state,
        algorithm="lloyd",
    )

    labels = kmeans.fit_predict(X)

    result = embeddings.copy()
    result["cluster"] = labels

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=str(EMBEDDING_FILE))
    parser.add_argument("--unweighted", action="store_true")  # default: level-weighted mean

    args = parser.parse_args()
    out_path = Path("data/category_embeddings.npy")
    # print(f"{out_path} alreaydy exists, skipping")
     
    embeddings = build_category_embeddings(load_category_taxonomy(), args.model, weighted=not args.unweighted)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_parquet(out_path, index=False)
    print(f"saved {len(embeddings)} embeddings "
        f"(dim {len(embeddings['embedding'].iloc[0])}) to {out_path}")

    clustered = cluster_embeddings(embeddings, k=50)
    clustered.drop(columns="embedding").to_parquet(PROCESSED_DIR / "category_clusters_K50.parquet", index=False)

    # readable listing: one block per cluster
    with open(BASE_PATH / "reports" / "clusters_K50.txt", "w", encoding="utf-8") as listing_file:
        for cluster_id, members in clustered.groupby("cluster"):
            listing_file.write(f"cluster {cluster_id} ({len(members)} categories): "
                            + "; ".join(members["category_name"]) + "\n\n")

if __name__ == "__main__":
    main()
