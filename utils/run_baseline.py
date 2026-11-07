"""Execute the baseline notebook in a fresh process, using the active Python profile.

The notebook remains the source of the dataset, verifier and inference logic.
The default phase is off, which runs only CPU data/mechanism checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("off", "calibration", "calibration_extension", "evaluation"), default="off")
    parser.add_argument("--run-id")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    os.environ["BASELINE_PHASE"] = args.phase
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.run_id:
        os.environ["BASELINE_RUN_ID"] = args.run_id
    if args.model_path:
        os.environ["QWEN3_MODEL_PATH"] = str(args.model_path.resolve())
    notebook_path = project_root / "00_Baseline_Diagnosis.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            print(f"BASELINE_CELL {index}", flush=True)
            exec(compile("".join(cell["source"]), f"{notebook_path}:cell_{index}", "exec"), namespace)
    status = namespace["baseline_status"]
    if args.phase != "off" and status.get("status") != "completed":
        raise SystemExit(f"Baseline phase did not complete: {status.get('status')}")
    print(json.dumps({"phase": args.phase, "status": status["status"],
                      "model_id": namespace["MODEL_ID"], "run_dir": str(namespace["run_dir"])}), flush=True)


if __name__ == "__main__":
    main()
