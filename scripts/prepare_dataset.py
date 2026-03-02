import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from framing_sensitivity.constants import DATASET_URL, FRAME_ORDER, N_PROMPTS
from framing_sensitivity.framing import build_framed_prompt


def download_source_csv(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATASET_URL) as response:
        destination.write_bytes(response.read())


def main() -> None:
    raw_path = REPO_ROOT / "data" / "raw" / "harmbench_behaviors_text_test.csv"
    originals_path = REPO_ROOT / "data" / "processed" / "original_prompts.csv"
    framed_path = REPO_ROOT / "data" / "processed" / "framed_prompts.csv"

    download_source_csv(raw_path)
    source_df = pd.read_csv(raw_path)
    if len(source_df) < N_PROMPTS:
        raise ValueError(f"Expected at least {N_PROMPTS} HarmBench rows but found {len(source_df)}.")
    selected_df = source_df.head(N_PROMPTS).copy()
    selected_df.insert(0, "source_row_index", range(len(selected_df)))
    selected_df.insert(0, "prompt_index", range(1, len(selected_df) + 1))
    selected_df = selected_df.rename(
        columns={
            "BehaviorID": "behavior_id",
            "Behavior": "original_prompt",
            "FunctionalCategory": "functional_category",
            "SemanticCategory": "semantic_category",
        }
    )
    selected_df = selected_df[
        [
            "prompt_index",
            "source_row_index",
            "behavior_id",
            "original_prompt",
            "functional_category",
            "semantic_category",
            "Tags",
            "ContextString",
        ]
    ].rename(columns={"Tags": "tags", "ContextString": "context_string"})

    originals_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(originals_path, index=False)

    framed_records = []
    for row in selected_df.itertuples(index=False):
        for frame_order, frame_name in enumerate(FRAME_ORDER, start=1):
            framed_records.append(
                {
                    "prompt_index": row.prompt_index,
                    "source_row_index": row.source_row_index,
                    "behavior_id": row.behavior_id,
                    "frame_order": frame_order,
                    "frame": frame_name,
                    "original_prompt": row.original_prompt,
                    "prompt_text": build_framed_prompt(row.original_prompt, frame_name),
                }
            )

    pd.DataFrame(framed_records).to_csv(framed_path, index=False)


if __name__ == "__main__":
    main()
