"""CLI 入口：python -m SAEGenRec.sid_builder <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.sid_builder.generate_indices import generate_indices
from SAEGenRec.sid_builder.rqkmeans import rqkmeans_constrained, rqkmeans_faiss, rqkmeans_plus
from SAEGenRec.sid_builder.rqvae import rqvae_train
from SAEGenRec.sid_builder.text2emb import text2emb

if __name__ == "__main__":
    fire.Fire(
        {
            "text2emb": text2emb,
            "rqvae_train": rqvae_train,
            "rqkmeans_faiss": rqkmeans_faiss,
            "rqkmeans_constrained": rqkmeans_constrained,
            "rqkmeans_plus": rqkmeans_plus,
            "generate_indices": generate_indices,
        }
    )
