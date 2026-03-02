import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from framing_sensitivity.constants import DECODING_PRESETS, FRAME_ORDER, GENERATION_DEFAULTS, MODEL_ID, MODEL_REVISION, N_PROMPTS
from framing_sensitivity.refusal import classify_refusal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/framed_prompts.csv",
        help="Path to the framed prompts CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/completions/completions.csv",
        help="Path to the completions CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompt/frame pairs that already exist in the output CSV.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=tuple(DECODING_PRESETS.keys()),
        default=list(DECODING_PRESETS.keys()),
        help="Decoding presets to run. Defaults to both main and appendix presets.",
    )
    return parser.parse_args()


def load_existing_pairs(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    existing_df = pd.read_csv(path)
    return set(zip(existing_df["decoding_preset"], existing_df["prompt_index"], existing_df["frame"]))


def build_model_inputs(tokenizer: AutoTokenizer, prompt_text: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids.to(device)


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    decoding_config: dict[str, object],
    device: torch.device,
) -> str:
    input_ids = build_model_inputs(tokenizer, prompt_text, device)
    generation_kwargs = {
        "input_ids": input_ids,
        "do_sample": bool(decoding_config["do_sample"]),
        "max_new_tokens": GENERATION_DEFAULTS["max_new_tokens"],
        "repetition_penalty": GENERATION_DEFAULTS["repetition_penalty"],
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if bool(decoding_config["do_sample"]):
        generation_kwargs["temperature"] = float(decoding_config["temperature"])
        generation_kwargs["top_p"] = GENERATION_DEFAULTS["top_p"]
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_df = pd.DataFrame([record])
    write_header = not path.exists()
    frame_df.to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    args = parse_args()
    input_path = REPO_ROOT / args.input
    output_path = REPO_ROOT / args.output

    prompts_df = pd.read_csv(input_path)
    expected_rows = N_PROMPTS * len(FRAME_ORDER)
    if len(prompts_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} framed prompts but found {len(prompts_df)}.")
    if prompts_df.duplicated(subset=["prompt_index", "frame"]).any():
        raise ValueError("Duplicate prompt/frame rows found in framed_prompts.csv.")
    completed_pairs = load_existing_pairs(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        output_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    total_jobs = len(prompts_df) * len(args.presets)
    progress = tqdm(total=total_jobs, desc="Running inference")
    for preset_name in args.presets:
        decoding_config = DECODING_PRESETS[preset_name]
        for row in prompts_df.itertuples(index=False):
            key = (preset_name, row.prompt_index, row.frame)
            if key in completed_pairs:
                progress.update(1)
                continue

            prompt_seed = int(decoding_config["seed"]) + (row.prompt_index * 10) + row.frame_order
            set_seed(prompt_seed)
            raw_output = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_text=row.prompt_text,
                decoding_config=decoding_config,
                device=device,
            )
            refusal_label, matched_pattern = classify_refusal(raw_output)
            append_record(
                output_path,
                {
                    "decoding_preset": preset_name,
                    "analysis_section": decoding_config["analysis_section"],
                    "prompt_index": row.prompt_index,
                    "source_row_index": row.source_row_index,
                    "behavior_id": row.behavior_id,
                    "frame_order": row.frame_order,
                    "frame": row.frame,
                    "prompt_text": row.prompt_text,
                    "raw_output": raw_output,
                    "completion": raw_output,
                    "refusal_label": refusal_label,
                    "matched_pattern": matched_pattern,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "temperature": float(decoding_config["temperature"]),
                    "top_p": GENERATION_DEFAULTS["top_p"],
                    "repetition_penalty": GENERATION_DEFAULTS["repetition_penalty"],
                    "max_new_tokens": GENERATION_DEFAULTS["max_new_tokens"],
                    "do_sample": bool(decoding_config["do_sample"]),
                    "seed": prompt_seed,
                    "base_seed": int(decoding_config["seed"]),
                },
            )
            progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()
