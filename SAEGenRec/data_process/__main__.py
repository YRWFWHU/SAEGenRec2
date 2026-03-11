"""CLI 入口：python -m SAEGenRec.data_process <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.data_process.convert_dataset import convert_dataset
from SAEGenRec.data_process.embed_text import embed_text
from SAEGenRec.data_process.image_downloader import download_images
from SAEGenRec.data_process.preprocess import preprocess
from SAEGenRec.data_process.visual_embed import extract_visual

if __name__ == "__main__":
    fire.Fire(
        {
            "preprocess": preprocess,
            "convert_dataset": convert_dataset,
            "download_images": download_images,
            "embed_text": embed_text,
            "extract_visual": extract_visual,
        }
    )
