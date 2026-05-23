#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportExplicitAny=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportArgumentType=false, reportPrivateImportUsage=false

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import DISTANCE_THRESHOLDS_KM, haversine_km
from src.skill_parser import COUNTRY_TO_REGION

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif"})

OUR_METHODS = [
    "direct_vlm",
    "cot_vlm",
    "geocot",
    "georeasoner",
    "gre_multistage",
    "skill_conditioned",
    "img2loc_rag",
]
ABLATION_METHODS = ["no_skill", "random_skill", "shuffled_order", "atomic_only", "composed_only"]
REGION_ORDER = ["europe", "asia", "north_america", "south_america", "africa", "oceania"]
GEORC_BASELINES = {
    "Human Expert Avg": 0.95,
    "Gemini-3-Pro": 0.904,
    "Gemini-2.5-Pro": 0.912,
    "GPT-5": 0.884,
    "GPT-4.1": 0.778,
    "Qwen2.5-VL-7B": 0.694,
    "Llama-3.2-11B": 0.584,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NeurIPS 2026 GeoSkill figures and tables")
    p.add_argument("--results-dir", type=str, default="experiments/full_100")
    p.add_argument("--ablation-dir", type=str, default="experiments/ablation")
    p.add_argument("--output-dir", type=str, default="figures")
    p.add_argument("--data-root", type=str, default="data/georc")
    return p.parse_args()


def _safe_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    return float("nan")


def _is_valid_coord(lat: Any, lng: Any) -> bool:
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lng_f):
        return False
    return -90 <= lat_f <= 90 and -180 <= lng_f <= 180


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Failed to parse JSON: {path} ({exc})")
        return None


def load_method_metrics(results_dir: Path, methods: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        obj = load_json(results_dir / method / "latest_metrics.json")
        if isinstance(obj, dict):
            out[method] = obj
    summary = load_json(results_dir / "summary_metrics.json")
    if isinstance(summary, dict):
        for method in methods:
            if method not in out and isinstance(summary.get(method), dict):
                out[method] = summary[method]
    return out


def load_method_predictions(results_dir: Path, methods: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        obj = load_json(results_dir / method / "latest_predictions.json")
        if isinstance(obj, list):
            out[method] = [x for x in obj if isinstance(x, dict)]
    return out


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def figure_main_results(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    methods = [m for m in OUR_METHODS if m in metrics]
    if not methods:
        warnings.warn("Skipping Figure 1: no metrics")
        return
    x = np.arange(len(methods))
    width = 0.38
    cvals = np.array([_safe_float(metrics[m].get("country_accuracy", np.nan)) for m in methods])
    rvals = np.array([_safe_float(metrics[m].get("continent_accuracy", np.nan)) for m in methods])
    pal = sns.color_palette("colorblind", n_colors=8)
    fig, ax = plt.subplots(figsize=(11, 5))
    bc = ax.bar(x - width / 2, cvals, width, label="Country accuracy", color=pal[0])
    br = ax.bar(x + width / 2, rvals, width, label="Continent accuracy", color=pal[2])
    for i, m in enumerate(methods):
        if m == "skill_conditioned":
            bc[i].set_facecolor(pal[1])
            bc[i].set_hatch("///")
            br[i].set_facecolor(pal[3])
            br[i].set_hatch("///")
    for i, (name, acc) in enumerate(GEORC_BASELINES.items()):
        ax.axhline(acc, color=pal[(i + 4) % len(pal)], linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(len(methods) - 0.45, acc + 0.005, f"{name}: {acc:.3f}", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 1: Main Results")
    ax.legend(loc="upper left")
    save_figure(fig, output_dir, "main_results")


def _method_distances(records: list[dict[str, Any]]) -> list[float]:
    dists: list[float] = []
    for rec in records:
        pred = rec.get("prediction", {})
        if not isinstance(pred, dict):
            continue
        plat = pred.get("predicted_lat")
        plng = pred.get("predicted_lng")
        glat = rec.get("ground_truth_lat")
        glng = rec.get("ground_truth_lng")
        if _is_valid_coord(plat, plng) and _is_valid_coord(glat, glng):
            dists.append(haversine_km(float(plat), float(plng), float(glat), float(glng)))
    return dists


def figure_distance_cdf(preds: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    methods = [m for m in OUR_METHODS if m in preds]
    if not methods:
        warnings.warn("Skipping Figure 2: no predictions")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pal = sns.color_palette("colorblind", n_colors=max(8, len(methods)))
    ok = False
    for i, m in enumerate(methods):
        d = sorted(_method_distances(preds[m]))
        if not d:
            continue
        y = np.arange(1, len(d) + 1) / len(d)
        ax.plot(d, y, color=pal[i], linewidth=2, label=m)
        ok = True
    if not ok:
        warnings.warn("Skipping Figure 2: no valid coordinates")
        plt.close(fig)
        return
    for t in DISTANCE_THRESHOLDS_KM:
        ax.axvline(t, color="gray", linestyle=":" if t != 750 else "--", linewidth=1)
        ax.text(t, 0.04, f"{t}km", rotation=90, fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Distance error (km, log scale)")
    ax.set_ylabel("Fraction of samples within distance")
    ax.set_title("Figure 2: Distance Error CDF")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, output_dir, "distance_cdf")


def _country_acc_by_region(records: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {r: [] for r in REGION_ORDER}
    for rec in records:
        gt_country = str(rec.get("ground_truth_country", "")).lower()
        reg = COUNTRY_TO_REGION.get(gt_country, "unknown")
        pred = rec.get("prediction", {})
        pc = str(pred.get("predicted_country", "")).lower() if isinstance(pred, dict) else ""
        if reg in buckets:
            buckets[reg].append(1.0 if pc == gt_country else 0.0)
    return {r: (float(np.mean(v)) if v else float("nan")) for r, v in buckets.items()}


def figure_region_heatmap(preds: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    methods = [m for m in OUR_METHODS if m in preds]
    if not methods:
        warnings.warn("Skipping Figure 3: no predictions")
        return
    mat = np.full((len(methods), len(REGION_ORDER)), np.nan, dtype=float)
    for i, m in enumerate(methods):
        row = _country_acc_by_region(preds[m])
        for j, reg in enumerate(REGION_ORDER):
            mat[i, j] = row[reg]
    if np.all(np.isnan(mat)):
        warnings.warn("Skipping Figure 3: empty matrix")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cmap = sns.diverging_palette(12, 133, as_cmap=True)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=REGION_ORDER,
        yticklabels=methods,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Country accuracy"},
    )
    ax.set_title("Figure 3: Region-wise Country Accuracy Heatmap")
    ax.set_xlabel("Region (ground truth)")
    ax.set_ylabel("Method")
    save_figure(fig, output_dir, "region_heatmap")


def figure_ablation(full: dict[str, dict[str, Any]], abl: dict[str, dict[str, Any]], output_dir: Path) -> None:
    if "skill_conditioned" not in full:
        warnings.warn("Skipping Figure 4: missing skill_conditioned full metrics")
        return
    variants = ["skill_conditioned"] + [m for m in ABLATION_METHODS if m in abl]
    if len(variants) <= 1:
        warnings.warn("Skipping Figure 4: no ablation metrics")
        return
    cvals: list[float] = []
    dvals: list[float] = []
    for v in variants:
        src = full["skill_conditioned"] if v == "skill_conditioned" else abl[v]
        cvals.append(_safe_float(src.get("country_accuracy", np.nan)))
        dvals.append(_safe_float(src.get("distance_error_km_mean_valid_only", np.nan)))
    x = np.arange(len(variants))
    pal = sns.color_palette("colorblind", n_colors=8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    b1 = axes[0].bar(x, cvals, color=pal[0])
    b1[0].set_color(pal[1])
    b1[0].set_hatch("///")
    axes[0].set_title("Country Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=25, ha="right")
    b2 = axes[1].bar(x, dvals, color=pal[2])
    b2[0].set_color(pal[3])
    b2[0].set_hatch("///")
    axes[1].set_title("Mean Distance Error (km)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants, rotation=25, ha="right")
    fig.suptitle("Figure 4: Ablation Results")
    save_figure(fig, output_dir, "ablation_results")


def _choose_skill_examples(records: list[dict[str, Any]], k: int = 4) -> list[dict[str, Any]]:
    cand: list[dict[str, Any]] = []
    for rec in records:
        pred = rec.get("prediction", {})
        if not isinstance(pred, dict):
            continue
        skills = pred.get("retrieved_skills")
        scene = pred.get("scene_description")
        if not isinstance(skills, list) or not skills:
            continue
        if not isinstance(scene, str) or not scene.strip():
            continue
        gt = str(rec.get("ground_truth_country", "")).lower()
        pc = str(pred.get("predicted_country", "")).lower()
        conf = _safe_float(pred.get("confidence", 0.0))
        score = (10 if gt == pc else 0) + min(len(skills), 5) + (0 if math.isnan(conf) else conf)
        rec["_score"] = score
        cand.append(rec)
    cand.sort(key=lambda x: _safe_float(x.get("_score", 0.0)), reverse=True)
    return cand[:k]


def _image_path(data_root: Path, rec: dict[str, Any]) -> Path:
    gid = str(rec.get("game_id", ""))
    rnd = int(rec.get("round", 1))
    return data_root / gid / f"{gid}_{rnd}.png"


def figure_skill_retrieval(preds: dict[str, list[dict[str, Any]]], output_dir: Path, data_root: Path) -> None:
    records = preds.get("skill_conditioned")
    if not records:
        warnings.warn("Skipping Figure 5: missing skill_conditioned predictions")
        return
    ex = _choose_skill_examples(records, k=4)
    if not ex:
        warnings.warn("Skipping Figure 5: no examples with scene+skills")
        return
    rows = len(ex)
    fig, axes = plt.subplots(rows, 4, figsize=(16, max(6, 3 * rows)))
    if rows == 1:
        axes = np.array([axes])
    for i, rec in enumerate(ex):
        pred = rec.get("prediction", {})
        if not isinstance(pred, dict):
            continue
        ax_img, ax_scene, ax_sk, ax_pred = axes[i]
        for ax in (ax_scene, ax_sk, ax_pred):
            ax.axis("off")
        ip = _image_path(data_root, rec)
        if ip.exists():
            try:
                ax_img.imshow(mpimg.imread(ip))
                ax_img.axis("off")
            except Exception:
                ax_img.text(0.5, 0.5, f"Image read failed\n{ip.name}", ha="center", va="center")
                ax_img.axis("off")
        else:
            ax_img.text(0.5, 0.5, f"Image not found\n{ip.name}", ha="center", va="center")
            ax_img.axis("off")
        scene = str(pred.get("scene_description", "")).strip()
        scene_preview = "\n".join(scene.splitlines()[:10])
        ax_scene.text(0.01, 0.99, f"Scene Description\n\n{scene_preview}", va="top", fontsize=8)
        lines: list[str] = []
        skills = pred.get("retrieved_skills", [])
        if isinstance(skills, list):
            for s in skills[:5]:
                if isinstance(s, dict):
                    txt = str(s.get("skill_text", ""))[:120]
                    hint = str(s.get("region_hint", "unknown"))
                    sc = _safe_float(s.get("score", np.nan))
                    lines.append(f"[{hint}|{sc:.2f}] {txt}")
        ax_sk.text(0.01, 0.99, "Retrieved Skills\n\n" + "\n".join(lines), va="top", fontsize=8)
        gt = str(rec.get("ground_truth_country", "")).upper()
        pc = str(pred.get("predicted_country", "")).upper()
        conf = _safe_float(pred.get("confidence", np.nan))
        ok = "✓" if gt.lower() == pc.lower() else "✗"
        ax_pred.text(0.01, 0.99, f"Final Prediction\n\nGT: {gt}\nPred: {pc} {ok}\nConf: {conf:.2f}", va="top", fontsize=10)
    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Scene Description")
    axes[0, 2].set_title("Retrieved Skills")
    axes[0, 3].set_title("Prediction")
    fig.suptitle("Figure 5: Skill Retrieval Visualization", y=1.02)
    save_figure(fig, output_dir, "skill_retrieval")


def figure_confusion_matrix(preds: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    records = preds.get("skill_conditioned")
    if not records:
        warnings.warn("Skipping Figure 6: missing skill_conditioned predictions")
        return
    gt = [str(r.get("ground_truth_country", "")).lower() for r in records]
    top20 = [c for c, _ in Counter(gt).most_common(20) if c]
    if len(top20) < 2:
        warnings.warn("Skipping Figure 6: insufficient country diversity")
        return
    idx = {c: i for i, c in enumerate(top20)}
    mat = np.zeros((len(top20), len(top20)), dtype=int)
    for r in records:
        g = str(r.get("ground_truth_country", "")).lower()
        pred = r.get("prediction", {})
        p = str(pred.get("predicted_country", "")).lower() if isinstance(pred, dict) else ""
        if g in idx and p in idx:
            mat[idx[g], idx[p]] += 1
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.upper() for c in top20],
        yticklabels=[c.upper() for c in top20],
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title("Figure 6: Country Confusion Matrix (Top-20 GT Countries)")
    ax.set_xlabel("Predicted country")
    ax.set_ylabel("Ground-truth country")
    save_figure(fig, output_dir, "confusion_matrix")


def _calibration_curve(records: list[dict[str, Any]], bins: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    confs: list[float] = []
    corr: list[float] = []
    for rec in records:
        pred = rec.get("prediction", {})
        if not isinstance(pred, dict):
            continue
        conf = _safe_float(pred.get("confidence", np.nan))
        if math.isnan(conf):
            continue
        conf = min(max(conf, 0.0), 1.0)
        gt = str(rec.get("ground_truth_country", "")).lower()
        pc = str(pred.get("predicted_country", "")).lower()
        confs.append(conf)
        corr.append(1.0 if gt == pc else 0.0)
    if not confs:
        return np.array([]), np.array([]), float("nan")
    conf_arr = np.array(confs)
    corr_arr = np.array(corr)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(conf_arr, edges, right=True) - 1, 0, bins - 1)
    x: list[float] = []
    y: list[float] = []
    ece = 0.0
    n = len(conf_arr)
    for b in range(bins):
        mask = ids == b
        if not np.any(mask):
            continue
        bconf = float(np.mean(conf_arr[mask]))
        bacc = float(np.mean(corr_arr[mask]))
        w = np.sum(mask) / n
        ece += w * abs(bacc - bconf)
        x.append(bconf)
        y.append(bacc)
    return np.array(x), np.array(y), float(ece)


def figure_calibration(preds: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    methods = [m for m in OUR_METHODS if m in preds]
    if not methods:
        warnings.warn("Skipping Figure 7: no predictions")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    pal = sns.color_palette("colorblind", n_colors=max(8, len(methods)))
    ok = False
    for i, m in enumerate(methods):
        x, y, ece = _calibration_curve(preds[m], bins=10)
        if x.size == 0:
            continue
        ax.plot(x, y, marker="o", linewidth=1.8, color=pal[i], label=f"{m} (ECE={ece:.3f})")
        ok = True
    if not ok:
        warnings.warn("Skipping Figure 7: no confidence values")
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Figure 7: Confidence Calibration")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_dir, "calibration")


def _format_val(v: float, percent: bool = False) -> str:
    if math.isnan(v):
        return "--"
    if percent:
        return f"{100.0 * v:.1f}\\%"
    return f"{v:.3f}"


def _rank_indices(values: list[float], higher: bool) -> tuple[int | None, int | None]:
    valid = [(i, v) for i, v in enumerate(values) if not math.isnan(v)]
    if not valid:
        return None, None
    valid.sort(key=lambda x: x[1], reverse=higher)
    return valid[0][0], (valid[1][0] if len(valid) > 1 else None)


def _decorate(val: str, idx: int, best: int | None, second: int | None) -> str:
    if best is not None and idx == best:
        return f"\\textbf{{{val}}}"
    if second is not None and idx == second:
        return f"\\underline{{{val}}}"
    return val


def table_main_results(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for m in OUR_METHODS:
        d = metrics.get(m, {})
        rows.append(
            {
                "method": m,
                "country": _safe_float(d.get("country_accuracy", np.nan)),
                "continent": _safe_float(d.get("continent_accuracy", np.nan)),
                "dist_median": _safe_float(d.get("distance_error_km_median", np.nan)),
                "valid": _safe_float(d.get("valid_coordinate_rate", np.nan)),
                "acc1": _safe_float(d.get("Acc@1km", np.nan)),
                "acc25": _safe_float(d.get("Acc@25km", np.nan)),
                "acc150": _safe_float(d.get("Acc@150km", np.nan)),
                "acc750": _safe_float(d.get("Acc@750km", np.nan)),
                "acc2500": _safe_float(d.get("Acc@2500km", np.nan)),
                "f1": _safe_float(d.get("expert_chain_token_f1", np.nan)),
                "halluc": _safe_float(d.get("heuristic_hallucination_rate", np.nan)),
            }
        )
    for name, acc in GEORC_BASELINES.items():
        rows.append(
            {
                "method": f"{name} (GeoRC)",
                "country": float(acc),
                "continent": float("nan"),
                "dist_median": float("nan"),
                "valid": float("nan"),
                "acc1": float("nan"),
                "acc25": float("nan"),
                "acc150": float("nan"),
                "acc750": float("nan"),
                "acc2500": float("nan"),
                "f1": float("nan"),
                "halluc": float("nan"),
            }
        )

    b_country, s_country = _rank_indices([r["country"] for r in rows], True)
    b_continent, s_continent = _rank_indices([r["continent"] for r in rows], True)
    b_dist_median, s_dist_median = _rank_indices([r["dist_median"] for r in rows], False)
    b_valid, s_valid = _rank_indices([r["valid"] for r in rows], True)
    b_acc1, s_acc1 = _rank_indices([r["acc1"] for r in rows], True)
    b_acc25, s_acc25 = _rank_indices([r["acc25"] for r in rows], True)
    b_acc150, s_acc150 = _rank_indices([r["acc150"] for r in rows], True)
    b_acc750, s_acc750 = _rank_indices([r["acc750"] for r in rows], True)
    b_acc2500, s_acc2500 = _rank_indices([r["acc2500"] for r in rows], True)
    b_f1, s_f1 = _rank_indices([r["f1"] for r in rows], True)
    b_halluc, s_halluc = _rank_indices([r["halluc"] for r in rows], False)

    lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccccccccc}",
        "\\toprule",
        "Method & Country & Continent & Dist median & Valid\\% & Acc@1km & Acc@25km & Acc@150km & Acc@750km & Acc@2500km & Expert-chain token F1 & Heuristic hallucination rate \\\\",
        "\\midrule",
    ]

    for i, r in enumerate(rows):
        country = _decorate(_format_val(r["country"]), i, b_country, s_country)
        continent = _decorate(_format_val(r["continent"]), i, b_continent, s_continent)
        dist_median = _decorate(_format_val(r["dist_median"]), i, b_dist_median, s_dist_median)
        valid = _decorate(_format_val(r["valid"], True), i, b_valid, s_valid)
        acc1 = _decorate(_format_val(r["acc1"]), i, b_acc1, s_acc1)
        acc25 = _decorate(_format_val(r["acc25"]), i, b_acc25, s_acc25)
        acc150 = _decorate(_format_val(r["acc150"]), i, b_acc150, s_acc150)
        acc750 = _decorate(_format_val(r["acc750"]), i, b_acc750, s_acc750)
        acc2500 = _decorate(_format_val(r["acc2500"]), i, b_acc2500, s_acc2500)
        f1 = _decorate(_format_val(r["f1"]), i, b_f1, s_f1)
        halluc = _decorate(_format_val(r["halluc"], True), i, b_halluc, s_halluc)
        lines.append(
            f"{r['method']} & {country} & {continent} & {dist_median} & {valid} & {acc1} & {acc25} & {acc150} & {acc750} & {acc2500} & {f1} & {halluc} " + "\\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Table 1: Main results on GeoRC benchmark with updated metric names and Acc@k thresholds. GeoRC baselines use paper-reported country accuracy; unavailable fields are --. Best is bold and second-best underlined.}",
            "\\label{tab:main_results}",
            "\\end{table*}",
        ]
    )
    (output_dir / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_ablation(full: dict[str, dict[str, Any]], abl: dict[str, dict[str, Any]], output_dir: Path) -> None:
    if "skill_conditioned" not in full:
        warnings.warn("Skipping Table 2: missing skill_conditioned full metrics")
        return
    if not any(m in abl for m in ABLATION_METHODS):
        warnings.warn("Skipping Table 2: no ablation metrics")
        return
    full_c = _safe_float(full["skill_conditioned"].get("country_accuracy", np.nan))
    rows = [("skill_conditioned", full["skill_conditioned"])]
    rows.extend([(m, abl[m]) for m in ABLATION_METHODS if m in abl])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & Country Acc & Continent Acc & Dist (median km) & $\\Delta$ Country Acc \\\\",
        "\\midrule",
    ]
    for name, m in rows:
        c = _safe_float(m.get("country_accuracy", np.nan))
        r = _safe_float(m.get("continent_accuracy", np.nan))
        d = _safe_float(m.get("distance_error_km_median", np.nan))
        delta = c - full_c if (not math.isnan(c) and not math.isnan(full_c)) else float("nan")
        dstr = "--" if math.isnan(d) else f"{d:.1f}"
        dlt = "--" if math.isnan(delta) else f"{delta:+.3f}"
        lines.append(f"{name} & {_format_val(c)} & {_format_val(r)} & {dstr} & {dlt} " + "\\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Table 2: Ablation results relative to full skill-conditioned model.}",
            "\\label{tab:ablation}",
            "\\end{table}",
        ]
    )
    (output_dir / "ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else root / args.results_dir
    ablation_dir = Path(args.ablation_dir) if Path(args.ablation_dir).is_absolute() else root / args.ablation_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root / args.output_dir
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_method_metrics(results_dir, OUR_METHODS)
    preds = load_method_predictions(results_dir, OUR_METHODS)
    abl_metrics = load_method_metrics(ablation_dir, ABLATION_METHODS)

    figure_main_results(metrics, output_dir)
    figure_distance_cdf(preds, output_dir)
    figure_region_heatmap(preds, output_dir)
    figure_ablation(metrics, abl_metrics, output_dir)
    figure_skill_retrieval(preds, output_dir, data_root)
    figure_confusion_matrix(preds, output_dir)
    figure_calibration(preds, output_dir)
    table_main_results(metrics, output_dir)
    table_ablation(metrics, abl_metrics, output_dir)

    print(f"Done. Figures/tables written to: {output_dir}")


if __name__ == "__main__":
    main()
