import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from framing_sensitivity.analysis import exact_mcnemar_test
from framing_sensitivity.constants import (
    BOOTSTRAP_CI_ALPHA,
    BOOTSTRAP_ITERATIONS,
    DECODING_PRESETS,
    FRAME_COLORS,
    FRAME_ORDER,
    MANUAL_AUDIT_SAMPLE_SIZE,
    MANUAL_AUDIT_SEED,
    N_PROMPTS,
)
from framing_sensitivity.refusal import classify_refusal


def render_figure_1(refusal_rates: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    error_lower = refusal_rates["refusal_rate"] - refusal_rates["refusal_rate_ci_low"]
    error_upper = refusal_rates["refusal_rate_ci_high"] - refusal_rates["refusal_rate"]
    ax.bar(
        refusal_rates["frame"],
        refusal_rates["refusal_rate"],
        color=[FRAME_COLORS[frame] for frame in refusal_rates["frame"]],
        yerr=np.vstack([error_lower, error_upper]),
        capsize=4,
        ecolor="#333333",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Refusal rate")
    ax.set_xlabel("Framing")
    ax.set_title("Figure 1. Refusal rate by framing")
    for bar, rate in zip(ax.patches, refusal_rates["refusal_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def render_figure_2(prompt_level_df: pd.DataFrame, destination: Path) -> None:
    heatmap_df = prompt_level_df[list(FRAME_ORDER)]
    fig, ax = plt.subplots(figsize=(6, 12))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(FRAME_ORDER)))
    ax.set_xticklabels([name.title() for name in FRAME_ORDER])
    yticks = list(range(0, len(heatmap_df), 10))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(index + 1) for index in yticks])
    ax.set_xlabel("Framing")
    ax.set_ylabel("Prompt index")
    ax.set_title("Figure 2. Prompt-level refusal consistency")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Refusal")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def bootstrap_prompt_metrics(prompt_level_df: pd.DataFrame, seed: int) -> dict[str, object]:
    frame_values = prompt_level_df[list(FRAME_ORDER)].to_numpy(dtype=float)
    unstable_values = prompt_level_df["unstable"].to_numpy(dtype=float)
    n_prompts = len(prompt_level_df)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n_prompts, size=(BOOTSTRAP_ITERATIONS, n_prompts))
    boot_frame_rates = frame_values[sample_indices].mean(axis=1)
    boot_fsi = unstable_values[sample_indices].mean(axis=1)

    lower_pct = 100 * (BOOTSTRAP_CI_ALPHA / 2)
    upper_pct = 100 * (1 - BOOTSTRAP_CI_ALPHA / 2)
    frame_ci = {}
    for frame_index, frame_name in enumerate(FRAME_ORDER):
        ci_low, ci_high = np.percentile(boot_frame_rates[:, frame_index], [lower_pct, upper_pct])
        frame_ci[frame_name] = {
            "refusal_rate_ci_low": float(ci_low),
            "refusal_rate_ci_high": float(ci_high),
        }
    fsi_ci_low, fsi_ci_high = np.percentile(boot_fsi, [lower_pct, upper_pct])
    return {
        "frame_ci": frame_ci,
        "fsi_ci_low": float(fsi_ci_low),
        "fsi_ci_high": float(fsi_ci_high),
    }


def render_figure_3(prompt_level_df: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(prompt_level_df["refusal_count"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color="#4c72b0", rwidth=0.9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Number of refusals across 3 framings")
    ax.set_ylabel("Prompt count")
    ax.set_title("Figure 3. Distribution of framing sensitivity")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def build_prompt_level_df(completions_df: pd.DataFrame) -> pd.DataFrame:
    prompt_level_df = (
        completions_df.pivot(index="source_row_index", columns="frame", values="refusal_label")
        .reindex(index=range(N_PROMPTS), columns=FRAME_ORDER)
        .fillna(0)
        .astype(int)
    )
    prompt_level_df.insert(0, "prompt_index", range(1, N_PROMPTS + 1))
    prompt_level_df.insert(0, "source_row_index", range(N_PROMPTS))
    prompt_level_df["refusal_count"] = prompt_level_df[list(FRAME_ORDER)].sum(axis=1)
    prompt_level_df["unstable"] = prompt_level_df[list(FRAME_ORDER)].nunique(axis=1).gt(1).astype(int)
    return prompt_level_df


def build_group_fsi(prompt_level_df: pd.DataFrame, preset_name: str) -> list[dict[str, object]]:
    records = []
    subgroup_specs = (
        ("first_half", 0, 49),
        ("second_half", 50, 99),
    )
    for subgroup_name, start_index, end_index in subgroup_specs:
        subgroup_df = prompt_level_df[
            prompt_level_df["source_row_index"].between(start_index, end_index, inclusive="both")
        ].copy()
        records.append(
            {
                "decoding_preset": preset_name,
                "subgroup": subgroup_name,
                "start_source_row_index": start_index,
                "end_source_row_index": end_index,
                "n_prompts": len(subgroup_df),
                "unstable_prompt_count": int(subgroup_df["unstable"].sum()),
                "fsi": float(subgroup_df["unstable"].mean()),
            }
        )
    return records


def build_manual_audit_sample(
    completions_df: pd.DataFrame,
    sample_path: Path,
    summary_path: Path,
) -> None:
    flagged_df = completions_df[
        (completions_df["decoding_preset"] == "deterministic_main") & (completions_df["refusal_label"] == 1)
    ].copy()
    if flagged_df.empty:
        pd.DataFrame(
            columns=[
                "audit_case_id",
                "decoding_preset",
                "prompt_index",
                "source_row_index",
                "frame",
                "matched_pattern",
                "raw_output",
                "manual_false_positive",
                "manual_notes",
            ]
        ).to_csv(sample_path, index=False)
        pd.DataFrame(
            [
                {
                    "sample_size_requested": MANUAL_AUDIT_SAMPLE_SIZE,
                    "sample_size_available": 0,
                    "sample_size_reviewed": 0,
                    "false_positive_count": 0,
                    "false_positive_rate": np.nan,
                }
            ]
        ).to_csv(summary_path, index=False)
        return

    sample_size = min(MANUAL_AUDIT_SAMPLE_SIZE, len(flagged_df))
    sample_df = flagged_df.sample(n=sample_size, random_state=MANUAL_AUDIT_SEED).sort_values(
        ["prompt_index", "frame_order"]
    )
    sample_df = sample_df[
        [
            "decoding_preset",
            "prompt_index",
            "source_row_index",
            "frame",
            "matched_pattern",
            "raw_output",
        ]
    ].copy()
    sample_df.insert(0, "audit_case_id", range(1, len(sample_df) + 1))
    sample_df["manual_false_positive"] = pd.NA
    sample_df["manual_notes"] = ""

    if sample_path.exists():
        existing_df = pd.read_csv(sample_path)
        merge_keys = ["decoding_preset", "prompt_index", "source_row_index", "frame", "matched_pattern", "raw_output"]
        reusable_columns = [
            column
            for column in ["manual_false_positive", "manual_notes"]
            if column in existing_df.columns
        ]
        if reusable_columns:
            sample_df = sample_df.merge(
                existing_df[merge_keys + reusable_columns],
                on=merge_keys,
                how="left",
                suffixes=("", "_existing"),
            )
            for column in reusable_columns:
                existing_column = f"{column}_existing"
                sample_df[column] = sample_df[column].where(
                    sample_df[column].notna(),
                    sample_df[existing_column],
                )
                sample_df = sample_df.drop(columns=[existing_column])
    sample_df.to_csv(sample_path, index=False)

    reviewed_df = sample_df.dropna(subset=["manual_false_positive"]).copy()
    reviewed_df["manual_false_positive"] = reviewed_df["manual_false_positive"].astype(float)
    false_positive_rate = (
        float(reviewed_df["manual_false_positive"].mean()) if not reviewed_df.empty else np.nan
    )
    summary_df = pd.DataFrame(
        [
            {
                "sample_size_requested": MANUAL_AUDIT_SAMPLE_SIZE,
                "sample_size_available": sample_size,
                "sample_size_reviewed": int(len(reviewed_df)),
                "false_positive_count": int(reviewed_df["manual_false_positive"].sum()) if not reviewed_df.empty else 0,
                "false_positive_rate": false_positive_rate,
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)


def main() -> None:
    completions_path = REPO_ROOT / "results" / "completions" / "completions.csv"
    analysis_dir = REPO_ROOT / "results" / "analysis"
    figures_dir = REPO_ROOT / "results" / "figures"
    manual_audit_sample_path = analysis_dir / "manual_flagged_audit_sample.csv"
    manual_audit_summary_path = analysis_dir / "manual_flagged_audit_summary.csv"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    completions_df = pd.read_csv(completions_path)
    expected_inferences = N_PROMPTS * len(FRAME_ORDER) * len(DECODING_PRESETS)
    if len(completions_df) != expected_inferences:
        raise ValueError(
            f"Expected {expected_inferences} completions but found {len(completions_df)}. "
            "Run the full inference pipeline before analysis."
        )
    if completions_df.duplicated(subset=["decoding_preset", "prompt_index", "frame"]).any():
        raise ValueError("Duplicate decoding_preset/prompt_index/frame rows found in completions.csv.")
    expected_presets = set(DECODING_PRESETS)
    actual_presets = set(completions_df["decoding_preset"].unique())
    if actual_presets != expected_presets:
        raise ValueError(f"Expected decoding presets {sorted(expected_presets)} but found {sorted(actual_presets)}.")
    expected_frames = set(FRAME_ORDER)
    actual_frames = set(completions_df["frame"].unique())
    if actual_frames != expected_frames:
        raise ValueError(f"Expected frames {sorted(expected_frames)} but found {sorted(actual_frames)}.")

    recomputed_labels = completions_df["raw_output"].fillna("").map(classify_refusal)
    recomputed_df = pd.DataFrame(recomputed_labels.tolist(), columns=["recomputed_refusal_label", "recomputed_matched_pattern"])
    if not recomputed_df["recomputed_refusal_label"].astype(int).equals(completions_df["refusal_label"].astype(int)):
        raise ValueError("Stored refusal labels do not match the rule-based classifier.")
    stored_patterns = completions_df["matched_pattern"].fillna("")
    if not recomputed_df["recomputed_matched_pattern"].fillna("").equals(stored_patterns):
        raise ValueError("Stored matched patterns do not match the rule-based classifier.")

    refusal_summary_records = []
    pairwise_results = []
    prompt_level_records = []
    subgroup_records = []
    fsi_summary_records = []
    summary = {
        "n_prompts": N_PROMPTS,
        "n_inferences_total": expected_inferences,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "preset_summaries": {},
    }

    primary_prompt_level_df = None
    primary_refusal_rates = None

    for preset_name, decoding_config in DECODING_PRESETS.items():
        preset_df = completions_df[completions_df["decoding_preset"] == preset_name].copy()
        if len(preset_df) != N_PROMPTS * len(FRAME_ORDER):
            raise ValueError(f"Preset {preset_name} does not contain the expected 300 completions.")

        refusal_rates = (
            preset_df.groupby("frame", sort=False)["refusal_label"]
            .agg(["mean", "sum"])
            .reindex(FRAME_ORDER)
            .reset_index()
            .rename(columns={"mean": "refusal_rate", "sum": "refusal_count"})
        )
        refusal_rates.insert(0, "analysis_section", decoding_config["analysis_section"])
        refusal_rates.insert(0, "decoding_preset", preset_name)

        prompt_level_df = build_prompt_level_df(preset_df)
        prompt_level_df.insert(0, "analysis_section", decoding_config["analysis_section"])
        prompt_level_df.insert(0, "decoding_preset", preset_name)
        prompt_level_records.extend(prompt_level_df.to_dict(orient="records"))
        bootstrap_metrics = bootstrap_prompt_metrics(prompt_level_df, seed=20250301 + len(fsi_summary_records))
        subgroup_rows = build_group_fsi(prompt_level_df, preset_name)

        preset_pairwise_results = []
        for frame_a, frame_b in [("direct", "fictional"), ("direct", "translation"), ("fictional", "translation")]:
            both = prompt_level_df[[frame_a, frame_b]]
            table = [
                [
                    int(((both[frame_a] == 1) & (both[frame_b] == 1)).sum()),
                    int(((both[frame_a] == 1) & (both[frame_b] == 0)).sum()),
                ],
                [
                    int(((both[frame_a] == 0) & (both[frame_b] == 1)).sum()),
                    int(((both[frame_a] == 0) & (both[frame_b] == 0)).sum()),
                ],
            ]
            statistic, p_value = exact_mcnemar_test(table)
            preset_pairwise_results.append(
                {
                    "decoding_preset": preset_name,
                    "analysis_section": decoding_config["analysis_section"],
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "both_refusal": table[0][0],
                    "a_only_refusal": table[0][1],
                    "b_only_refusal": table[1][0],
                    "both_non_refusal": table[1][1],
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
        pairwise_results.extend(preset_pairwise_results)

        for record in refusal_rates.to_dict(orient="records"):
            record.update(bootstrap_metrics["frame_ci"][record["frame"]])
            refusal_summary_records.append(record)

        fsi_record = {
            "decoding_preset": preset_name,
            "analysis_section": decoding_config["analysis_section"],
            "n_prompts": int(len(prompt_level_df)),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "fsi": float(prompt_level_df["unstable"].mean()),
            "fsi_ci_low": bootstrap_metrics["fsi_ci_low"],
            "fsi_ci_high": bootstrap_metrics["fsi_ci_high"],
        }
        fsi_summary_records.append(fsi_record)

        subgroup_records_with_ci = []
        for subgroup_record in subgroup_rows:
            subgroup_df = prompt_level_df[
                prompt_level_df["source_row_index"].between(
                    subgroup_record["start_source_row_index"],
                    subgroup_record["end_source_row_index"],
                    inclusive="both",
                )
            ].copy()
            subgroup_bootstrap = bootstrap_prompt_metrics(
                subgroup_df,
                seed=MANUAL_AUDIT_SEED + len(subgroup_records_with_ci) + len(subgroup_records),
            )
            subgroup_record["analysis_section"] = decoding_config["analysis_section"]
            subgroup_record["bootstrap_iterations"] = BOOTSTRAP_ITERATIONS
            subgroup_record["fsi_ci_low"] = subgroup_bootstrap["fsi_ci_low"]
            subgroup_record["fsi_ci_high"] = subgroup_bootstrap["fsi_ci_high"]
            subgroup_records_with_ci.append(subgroup_record)
        subgroup_records.extend(subgroup_records_with_ci)

        preset_summary = {
            "analysis_section": decoding_config["analysis_section"],
            "n_inferences": int(len(preset_df)),
            "temperature": float(preset_df["temperature"].iloc[0]),
            "seed": int(preset_df["base_seed"].iloc[0]),
            "framing_sensitivity_index": float(prompt_level_df["unstable"].mean()),
            "framing_sensitivity_index_ci_low": bootstrap_metrics["fsi_ci_low"],
            "framing_sensitivity_index_ci_high": bootstrap_metrics["fsi_ci_high"],
            "unstable_prompt_count": int(prompt_level_df["unstable"].sum()),
            "refusal_rates": {
                row.frame: float(row.refusal_rate) for row in refusal_rates.itertuples(index=False)
            },
            "fsi_subgroups": {
                record["subgroup"]: {
                    "fsi": record["fsi"],
                    "fsi_ci_low": record["fsi_ci_low"],
                    "fsi_ci_high": record["fsi_ci_high"],
                }
                for record in subgroup_records_with_ci
            },
        }
        summary["preset_summaries"][preset_name] = preset_summary

        if preset_name == "deterministic_main":
            primary_prompt_level_df = prompt_level_df.copy()
            primary_refusal_rates = pd.DataFrame(
                [
                    {
                        **record,
                    }
                    for record in refusal_summary_records
                    if record["decoding_preset"] == "deterministic_main"
                ]
            )

    refusal_summary_df = pd.DataFrame(refusal_summary_records)
    refusal_summary_df.to_csv(analysis_dir / "refusal_summary.csv", index=False)

    fsi_summary_df = pd.DataFrame(fsi_summary_records)
    fsi_summary_df.to_csv(analysis_dir / "fsi_summary.csv", index=False)

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(analysis_dir / "pairwise_mcnemar.csv", index=False)

    prompt_level_outcomes_df = pd.DataFrame(prompt_level_records)
    prompt_level_outcomes_df.to_csv(analysis_dir / "prompt_level_outcomes.csv", index=False)

    subgroup_fsi_df = pd.DataFrame(subgroup_records)
    subgroup_fsi_df.to_csv(analysis_dir / "fsi_subgroups.csv", index=False)

    build_manual_audit_sample(completions_df, manual_audit_sample_path, manual_audit_summary_path)
    manual_audit_summary = pd.read_csv(manual_audit_summary_path).iloc[0]
    manual_audit_summary = manual_audit_summary.where(pd.notna(manual_audit_summary), None)
    summary["manual_audit"] = manual_audit_summary.to_dict()

    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if primary_prompt_level_df is None or primary_refusal_rates is None:
        raise ValueError("Missing deterministic_main analysis outputs.")
    render_figure_1(primary_refusal_rates, figures_dir / "figure_1_refusal_rate.png")
    render_figure_2(primary_prompt_level_df, figures_dir / "figure_2_prompt_level_consistency.png")
    render_figure_3(primary_prompt_level_df, figures_dir / "figure_3_framing_sensitivity_histogram.png")


if __name__ == "__main__":
    main()
