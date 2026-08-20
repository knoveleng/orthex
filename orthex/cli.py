from __future__ import annotations

import argparse
from pathlib import Path

from orthex import config as config_module
from orthex import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orthex")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        dest="configs",
        help="YAML config path; repeatable, later files deep-merge onto earlier ones",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Dotted-path config override, repeatable -- the one mechanism for overriding any field "
        "(e.g. --set model.id=Qwen/Qwen2.5-3B-Instruct --set model.architecture_adapter=qwen2)",
    )
    parser.add_argument("--output-dir", default=None, help="Overrides export.output_dir/<model-basename>")
    args = parser.parse_args(argv)

    raw = config_module.load_config_dict(*args.configs)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    raw = config_module.apply_dotted_overrides(raw, overrides)
    cfg = config_module.build_config(raw)

    output_dir = args.output_dir or str(Path(cfg.export.output_dir) / Path(cfg.model.id).name)
    result = pipeline.run(cfg, output_dir)
    print(f"Wrote {output_dir}/evaluation_report.json")
    print(f"  refusal_rate: {result['evaluation']['refusal_rate']}")
    print(f"  perplexity:   {result['evaluation']['perplexity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
