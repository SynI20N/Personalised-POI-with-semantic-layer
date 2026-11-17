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


def build_category_embeddings(taxonomy, model_name=DEFAULT_MODEL):
    """Embedding table: category_id, category_name, full_category_path, embedding."""
    table = category_texts(taxonomy)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=str(EMBEDDING_FILE))
    args = parser.parse_args()

    embeddings = build_category_embeddings(load_category_taxonomy(), args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_parquet(out_path, index=False)
    print(f"saved {len(embeddings)} embeddings "
          f"(dim {len(embeddings['embedding'].iloc[0])}) to {out_path}")


if __name__ == "__main__":
    main()
