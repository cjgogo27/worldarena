from __future__ import annotations

import json
from pathlib import Path


OUT = Path("/data/alice/cjtest/VideoX-Fun/six_model_available_comparison.md")

MODELS = {
    "ABot": {
        "standard": "/data/alice/cjtest/model_repros/worldarena_abot_public/metrics_output_test10_t1/generated_test10_t1_results.json",
        "vlm": "/data/alice/cjtest/model_repros/worldarena_abot_public/output_VLM/abot_public_test10_t1/abot_public_test10_t1_summary_val_all_intern.json",
    },
    "ABot(seedvr)": {
        "standard": "/data/alice/cjtest/model_repros/worldarena_abot_public/metrics_output_test10_t1_seedvr/generated_test10_t1_seedvr_results.json",
        "vlm": "/data/alice/cjtest/model_repros/worldarena_abot_public/output_VLM/abot_public_test10_t1_seedvr/abot_public_test10_t1_seedvr_summary_val_all_intern.json",
    },
    "Wan2.1": {
        "standard": "/data/alice/cjtest/model_repros/worldarena_wan_public/metrics_output_test10_t1/generated_test10_t1_results.json",
        "vlm": "/data/alice/cjtest/model_repros/worldarena_wan_public/output_VLM/wan_public_test10_t1/wan_public_test10_t1_summary_val_all_intern.json",
    },
    "Wan2.1(seedvr)": {
        "standard": "/data/alice/cjtest/model_repros/worldarena_wan_public/metrics_output_test10_t1_seedvr/generated_test10_t1_seedvr_results.json",
        "vlm": "/data/alice/cjtest/model_repros/worldarena_wan_public/output_VLM/wan_public_test10_t1_seedvr/wan_public_test10_t1_seedvr_summary_val_all_intern.json",
    },
    "SFT-Wan2.1": {
        "standard": "/data/alice/cjtest/VideoX-Fun/metrics_output_ckpt300_test10/eval_ckpt300_test10_generated_results.json",
        "vlm": "/data/alice/cjtest/VideoX-Fun/output_VLM_ckpt300_test10/videoxfun_ckpt300_test10/videoxfun_ckpt300_test10_summary_val_all_intern.json",
    },
    "SFT-Wan2.1(seedvr)": {
        "standard": "/data/alice/cjtest/VideoX-Fun/metrics_output_ckpt2200_seedvr_test10/eval_ckpt2200_seedvr_test10_generated_results.json",
        "vlm": "/data/alice/cjtest/VideoX-Fun/output_VLM_ckpt2200_seedvr_test10/videoxfun_ckpt2200_seedvr_test10/videoxfun_ckpt2200_seedvr_test10_summary_val_all_intern.json",
    },
}

STANDARD_METRICS = [
    ("image_quality", "Image Quality"),
    ("aesthetic_quality", "Aesthetic Quality"),
    ("background_consistency", "Background Consistency"),
    ("dynamic_degree", "Dynamic Degree"),
    ("flow_score", "Flow Score"),
    ("subject_consistency", "Subject Consistency"),
]

VLM_METRICS = [
    ("Interaction_Quality", "Interaction Quality"),
    ("Perspectivity", "Perspectivity"),
    ("Instruction_Following", "Instruction Following"),
]


def read_standard(path: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    values: dict[str, float] = {}
    for key, _ in STANDARD_METRICS:
        metric = data.get(key)
        if not (isinstance(metric, list) and metric):
            continue
        if len(metric) > 1 and isinstance(metric[1], list):
            normalized = [
                item.get("video_results_normalized")
                for item in metric[1]
                if isinstance(item, dict) and isinstance(item.get("video_results_normalized"), (int, float))
            ]
            if normalized:
                values[key] = (sum(normalized) / len(normalized)) * 100
                continue
        if isinstance(metric[0], (int, float)):
            values[key] = metric[0] * 100
    return values


def read_vlm(path: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    buckets: dict[str, list[float]] = {key: [] for key, _ in VLM_METRICS}
    for item in data:
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        for key in buckets:
            value = metrics.get(key, {}).get("score_normalized")
            if isinstance(value, (int, float)):
                buckets[key].append(value * 100)
    return {key: sum(vals) / len(vals) for key, vals in buckets.items() if vals}


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    values: dict[str, dict[str, float]] = {}
    availability: dict[str, dict[str, bool]] = {}
    for name, paths in MODELS.items():
        row = {}
        row.update(read_standard(paths["standard"]))
        row.update(read_vlm(paths["vlm"]))
        values[name] = row
        availability[name] = {
            "standard": Path(paths["standard"]).exists(),
            "vlm": Path(paths["vlm"]).exists(),
        }

    lines: list[str] = []
    lines.append("# 六模型 WorldArena 评测对比（现有结果先填，缺失留空）")
    lines.append("")
    lines.append("> 分数统一换算为 0-100。空白表示该项结果文件尚未生成或尚未确认。当前已填结果多为已有 10-sample 结果；后续按用户要求改为每类 50 个样本补评。")
    lines.append("")
    headers = ["Metric", *MODELS.keys()]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---", *["---:" for _ in MODELS]]) + " |")
    for key, label in [*STANDARD_METRICS, *VLM_METRICS]:
        cells = [label]
        for name in MODELS:
            cells.append(fmt(values[name].get(key)))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 结果文件状态")
    lines.append("")
    lines.append("| Model | Standard metrics | VLM judge |")
    lines.append("| --- | --- | --- |")
    for name, paths in MODELS.items():
        std = "✅" if availability[name]["standard"] else ""
        vlm = "✅" if availability[name]["vlm"] else ""
        lines.append(f"| {name} | {std} `{paths['standard']}` | {vlm} `{paths['vlm']}` |")

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
