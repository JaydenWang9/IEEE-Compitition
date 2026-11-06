"""Build the selected state predictions and assemble a local Kaggle ZIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OSCAR = ROOT / "experiments/trafficflowbench_local_20261102"
TASK1 = ROOT / "experiments/task1/best_road_combination_20261026"
STATE_SCRIPT = OSCAR / "task3/packages/severe_mixture/apply_residual.py"
QUEUE = ROOT / "predictions/task2"
FROZEN = OSCAR / "task4/frozen_reference_path_flow.csv"
EXPECTED = {"state": 6735795, "queue": 174000, "odme": 70708}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def correct_state(data, base, output, workers):
    files = [(split, path) for split in ("validation", "private")
             for path in sorted((base / split).glob("*.csv"))]
    if len(files) != 20:
        raise ValueError(f"Expected 20 base state CSVs, got {len(files)}")

    def build(item):
        split, source = item
        target = output / split / source.name
        temporary = target.with_suffix(".csv.tmp")
        subprocess.run([
            sys.executable, str(ROOT / "tools/portable_entry.py"), str(STATE_SCRIPT),
            "--data-root", str(data), "--state-csv", str(source), "--panel", source.stem,
            "--split", split, "--output", str(temporary),
        ], check=True)
        temporary.replace(target)
        print(f"State part ready: {split}/{source.name}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(build, files))


def assemble(data, state, odme, output):
    import pandas as pd

    sys.path.insert(0, str(TASK1 / "reproduction"))
    from merge_submission import merge

    od_keys = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]
    templates = sorted(data.glob("task4/*/validation/sample_submission_path_flow.csv"))
    templates += sorted(data.glob("task4/*/private/sample_submission_path_flow.csv"))
    expected_od = pd.concat([pd.read_csv(path, usecols=od_keys) for path in templates])
    actual_od = pd.read_csv(odme, usecols=od_keys)
    pd.testing.assert_series_equal(
        expected_od[od_keys].value_counts().sort_index(),
        actual_od[od_keys].value_counts().sort_index(),
    )
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / "submission.csv"
    checks = merge(
        data, state, [QUEUE / "queue-validation.csv", QUEUE / "queue-private.csv"],
        [odme], candidate,
    )
    if checks["counts"] != EXPECTED or checks["rows"] != sum(EXPECTED.values()):
        raise ValueError(f"Wrong row counts: {checks['counts']}")

    archive = output / "submission.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
        z.write(candidate, arcname="submission.csv")
    with ZipFile(temporary) as z:
        if z.namelist() != ["submission.csv"] or z.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    temporary.replace(archive)
    report = {
        "passed": True,
        "rows": checks["rows"],
        "counts": checks["counts"],
        "task3_uses_state_rows": True,
        "odme_source_sha256": digest(odme),
        "csv_sha256": checks["sha256"],
        "zip_sha256": digest(archive),
        "zip_bytes": archive.stat().st_size,
        "official_score": None,
    }
    (output / "build_acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-state-dir", type=Path, help="Apply Oscar correction to frozen base CSVs")
    source.add_argument("--state-dir", type=Path, help="Use existing Oscar-corrected state CSVs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--task4", type=Path, default=FROZEN, help="Optional approved keyed Task4 replacement")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    state = args.state_dir
    if args.base_state_dir is not None:
        state = args.output_dir / "state"
        if args.base_state_dir.resolve() == state.resolve():
            parser.error("The input and output state directories must differ")
        correct_state(args.data_root, args.base_state_dir, state, args.workers)
    assemble(args.data_root, state, args.task4, args.output_dir)


if __name__ == "__main__":
    main()
