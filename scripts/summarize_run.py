#!/usr/bin/env python3
"""Globs ./out/*/evaluation_report.json and prints a Markdown summary table
across all models in a run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./out")
    args = parser.parse_args()

    rows = []
    for report_path in sorted(Path(args.out_dir).glob("*/evaluation_report.json")):
        data = json.loads(report_path.read_text())
        rr = data["evaluation"]["refusal_rate"]
        pp = data["evaluation"]["perplexity"]
        sel = data["selected_candidate"]
        rows.append(
            {
                "model": Path(data["model_id"]).name,
                "adapter": data["architecture_adapter"],
                "selected": f"{sel['layer']}/{sel['site']}",
                "refusal_rate": f"{rr['pre']:.2f} -> {rr['post']:.2f} ({rr['delta']:+.2f})",
                "perplexity": f"{pp['pre']:.2f} -> {pp['post']:.2f} ({pp['delta']:+.2f})",
            }
        )

    header = "| model | adapter | selected layer/site | refusal_rate pre->post (delta) | perplexity pre->post (delta) |"
    sep = "|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['model']} | {r['adapter']} | {r['selected']} | {r['refusal_rate']} | {r['perplexity']} |")


if __name__ == "__main__":
    main()
