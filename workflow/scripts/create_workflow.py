#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflow.config import save_workflow_yaml


def main():
    parser = argparse.ArgumentParser(description="Create a new workflow")
    parser.add_argument("--name", type=str, required=True, help="Workflow name")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for workflows (default: current directory)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Workflow description",
    )
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path.cwd()
    wf_dir = root / args.name
    configs_dir = wf_dir / "configs"

    if wf_dir.exists():
        print(f"Error: {wf_dir} already exists")
        sys.exit(1)

    wf_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(exist_ok=True)

    wf_data = {
        "name": args.name,
        "description": args.description,
        "infrastructure": {},
        "stages": [],
    }
    save_workflow_yaml(wf_data, wf_dir / "workflow.yaml")
    print(f"Created workflow: {wf_dir}")


if __name__ == "__main__":
    main()
