#!/usr/bin/env python3
"""
Auto-fill ablation table in paper once all ablation variants complete.
Run after ablation finishes: python scripts/fill_ablation_table.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "neurips_2026.tex"
ABLATION_DIR = ROOT / "experiments" / "ablation"

VARIANTS = ["no_skill", "random_skill", "shuffled_order", "atomic_only", "composed_only"]
FULL_MODEL = {
    "country_accuracy": 0.44,
    "continent_accuracy": 0.83,
    "distance_error_km_median": 873.79,
}

DISPLAY_NAMES = {
    "no_skill": "No Skill (2-pass only)",
    "random_skill": "Random Skills",
    "shuffled_order": "Shuffled Order",
    "atomic_only": "Atomic Skills Only",
    "composed_only": "Composed Skills Only",
}


def load_metrics(variant: str) -> dict | None:
    path = ABLATION_DIR / variant / "latest_metrics.json"
    if not path.exists():
        print(f"  [MISSING] {variant}: {path}")
        return None
    return json.loads(path.read_text())


def fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"


def build_table_row(name: str, m: dict, full_country: float) -> str:
    ca = m.get("country_accuracy", 0.0)
    ra = m.get("continent_accuracy", 0.0)
    dm = m.get("distance_error_km_median", 0.0)
    delta = ca - full_country
    delta_str = f"${'+' if delta >= 0 else ''}{delta:.2f}$"
    return f"{name} & {fmt(ca)} & {fmt(ra)} & {round(dm)} & {delta_str} \\\\"


def main():
    print("Loading ablation results...")
    results = {}
    missing = []
    for v in VARIANTS:
        m = load_metrics(v)
        if m is None:
            missing.append(v)
        else:
            results[v] = m
            print(f"  [OK] {v}: country={m['country_accuracy']:.3f}, continent={m['continent_accuracy']:.3f}, dist_med={m['distance_error_km_median']:.0f}")

    if missing:
        print(f"\nWARNING: Missing variants: {missing}")
        print("Filling available variants only...")

    full_ca = FULL_MODEL["country_accuracy"]
    full_ra = FULL_MODEL["continent_accuracy"]
    full_dm = round(FULL_MODEL["distance_error_km_median"])

    full_row = f"Full Model (Ours) & {full_ca:.2f} & {full_ra:.2f} & {full_dm} & --- \\\\"

    variant_rows = []
    for v in VARIANTS:
        if v in results:
            row = build_table_row(DISPLAY_NAMES[v], results[v], full_ca)
        else:
            row = f"{DISPLAY_NAMES[v]} & \\multicolumn{{3}}{{c}}{{\\textit{{(pending)}}}} & --- \\\\"
        variant_rows.append(row)

    new_table_body = (
        full_row
        + "\n\\midrule\n"
        + "\n".join(variant_rows)
    )

    content = PAPER.read_text()

    old_start = "Full Model (Ours) &"
    old_end_marker = "Composed Skills Only"

    start_idx = content.find(old_start)
    if start_idx == -1:
        print("ERROR: Could not find ablation table start marker in paper")
        sys.exit(1)

    end_idx = content.find(old_end_marker, start_idx)
    if end_idx == -1:
        print("ERROR: Could not find ablation table end marker in paper")
        sys.exit(1)

    end_line = content.find("\n", end_idx)
    old_block = content[start_idx:end_line]

    print(f"\nReplacing:\n{old_block}\n\nWith:\n{new_table_body}")
    content = content[:start_idx] + new_table_body + content[end_line:]
    PAPER.write_text(content)
    print(f"\nPaper updated: {PAPER}")

    if "no_skill" in results:
        ns = results["no_skill"]
        ns_ca = ns["country_accuracy"]
        ns_errors = ns.get("num_errors", 28)

    print("\nRecompiling paper...")
    result = subprocess.run(
        ["tectonic", "neurips_2026.tex"],
        cwd=PAPER.parent,
        capture_output=True,
        text=True,
        env={"PATH": "/home/r1/.local/bin:/usr/bin:/bin", "HOME": "/home/r1"},
    )
    if result.returncode == 0:
        print("SUCCESS: neurips_2026.pdf compiled clean")
    else:
        print(f"COMPILE ERROR:\n{result.stderr}")

    print("\nDone.")


if __name__ == "__main__":
    main()
