#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflow.config import load_workflow_yaml, load_stage_toml, resolve_placeholders, save_stage_toml
from workflow.logger import EventQueue, WorkflowLogger
from workflow.models import WorkflowDefinition
from workflow.stages.preprocess import PreprocessExecutor
from workflow.stages.train import TrainExecutor


def main():
    parser = argparse.ArgumentParser(description="Run workflow stages step by step")
    parser.add_argument(
        "--workflow-dir",
        type=str,
        required=True,
        help="Path to the workflow directory",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage ID to execute (e.g. preprocess_s1, train_s1)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory (for stages after the first)",
    )
    args = parser.parse_args()

    wf_dir = Path(args.workflow_dir).resolve()
    wf_file = wf_dir / "workflow.yaml"
    wf_data = load_workflow_yaml(wf_file)
    wf = WorkflowDefinition(**wf_data)

    ordered = wf.topological_order()
    stage_map = {s.id: s for s in ordered}

    target_id = args.stage
    if target_id not in stage_map:
        print(f"Error: stage '{target_id}' not found. Available: {list(stage_map.keys())}")
        sys.exit(1)

    target_idx = next(i for i, s in enumerate(ordered) if s.id == target_id)
    stages_to_run = ordered[: target_idx + 1]

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        from datetime import datetime
        runs_dir = wf_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"
    eq = EventQueue()
    logger = WorkflowLogger(log_file, eq)

    stage_outputs: dict[str, dict[str, str]] = {}

    for stage in stages_to_run:
        if stage.id != target_id:
            stage_dir = run_dir / stage.id
            if not stage_dir.exists():
                print(f"Error: prerequisite stage dir not found: {stage_dir}")
                print(f"Run stage '{stage.id}' first.")
                sys.exit(1)

            output_dir = stage_dir / "output"
            if stage.type == "preprocess":
                dataset_dir = stage_dir / "post_image_dataset"
                stage_outputs[stage.id] = {"dataset_dir": str(dataset_dir)}
            elif stage.type == "train":
                safetensors = sorted(output_dir.glob("*.safetensors"))
                safetensors_path = str(safetensors[-1]) if safetensors else ""
                stage_outputs[stage.id] = {
                    "safetensors_path": safetensors_path,
                    "checkpoint_dir": str(output_dir),
                }
            print(f"[skip] {stage.id} — outputs loaded from {stage_dir}")
            continue

        config_path = wf_dir / "configs" / stage.config_file
        raw_config = load_stage_toml(config_path)
        resolved = resolve_placeholders(raw_config, stage_outputs)

        stage_dir = run_dir / stage.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        save_stage_toml(resolved, stage_dir / "config.toml")

        infra = wf.infrastructure or {}
        if stage.type == "preprocess":
            executor = PreprocessExecutor(stage.id, resolved, stage_dir, infra)
        elif stage.type == "train":
            executor = TrainExecutor(stage.id, resolved, stage_dir, infra)
        else:
            print(f"Error: unknown stage type '{stage.type}'")
            sys.exit(1)

        logger.workflow_start(1)
        logger.stage_start(stage.id, stage.type)

        print(f"\n{'='*60}")
        print(f"Executing: {stage.id} ({stage.type})")
        print(f"Stage dir: {stage_dir}")
        print(f"{'='*60}\n")

        def on_stdout(sid: str, line: str) -> None:
            logger.info(sid, line)
            print(f"[{sid}] {line}")

        result = executor.execute(on_stdout=on_stdout)

        if result.success:
            stage_outputs[stage.id] = result.outputs
            logger.stage_end(stage.id, "ok")
            print(f"\n[OK] {stage.id} completed successfully")
            print(f"Outputs: {result.outputs}")
        else:
            logger.stage_end(stage.id, f"error: {result.error}")
            print(f"\n[FAIL] {stage.id} failed: {result.error}")
            sys.exit(1)

    logger.workflow_end("ok")
    print(f"\nRun directory: {run_dir}")
    print(f"Stage outputs: {stage_outputs}")


if __name__ == "__main__":
    main()
