#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflow.config import load_workflow_yaml, save_workflow_yaml
from workflow.logger import EventQueue
from workflow.models import WorkflowDefinition
from workflow.scheduler import WorkflowScheduler


def main():
    parser = argparse.ArgumentParser(description="Run a workflow")
    parser.add_argument(
        "--workflow-dir",
        type=str,
        required=True,
        help="Path to the workflow directory (containing workflow.yaml)",
    )
    parser.add_argument(
        "--clear-runs",
        action="store_true",
        help="Remove all previous run directories before executing",
    )
    args = parser.parse_args()

    wf_dir = Path(args.workflow_dir).resolve()
    wf_file = wf_dir / "workflow.yaml"
    if not wf_file.exists():
        print(f"Error: {wf_file} not found")
        sys.exit(1)

    if args.clear_runs:
        runs_dir = wf_dir / "runs"
        if runs_dir.exists():
            import shutil
            shutil.rmtree(runs_dir)
            print(f"Cleared {runs_dir}")

    wf_data = load_workflow_yaml(wf_file)
    wf = WorkflowDefinition(**wf_data)
    eq = EventQueue()
    scheduler = WorkflowScheduler(wf_dir, wf, eq)

    print(f"Running workflow: {wf.name} ({len(wf.stages)} stages)")
    success = scheduler.run()
    if success:
        print("Workflow completed successfully")
    else:
        print("Workflow failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
