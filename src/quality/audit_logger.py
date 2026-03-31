"""Structured audit logger for DCLO pipeline runs.

Provides SHA-256 checksums for input/output files, pipeline run manifests,
and stage-level provenance records. Designed to meet academic replicability
standards per Christensen & Miguel (2018) transparency principles and
OECD/JEL data-citation guidance.

Key capabilities:
- Input file integrity verification (SHA-256)
- Row-level accounting at each pipeline stage (rows in, rows dropped, rows out)
- Deterministic seed recording for all stochastic operations
- Full config snapshot for exact replication
- Output file checksums for downstream verification
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_checksum(file_path: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot checksum missing file: {file_path}")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> Optional[str]:
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_installed_packages() -> Dict[str, str]:
    """Return dict of installed package names to versions for key dependencies."""
    packages = {}
    for pkg_name in ["pandas", "numpy", "scipy", "statsmodels", "pyyaml", "streamlit", "plotly"]:
        try:
            mod = __import__(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return packages


class AuditLogger:
    """Records pipeline provenance for a single run.

    Usage:
        audit = AuditLogger(pipeline_name="build_dclo_local")
        audit.record_config(config_dict)
        audit.record_input("nafis", "/path/to/nafis.csv")
        audit.start_stage("aggregate_nafis")
        audit.record_stage_io("aggregate_nafis", rows_in=500, rows_out=480, rows_dropped=20, drop_reasons={"null_year": 15, "null_state": 5})
        ...
        audit.record_output("dclo_state_year.csv", "/path/to/output.csv")
        audit.write_manifest("/path/to/manifest.json")
    """

    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.manifest: Dict[str, Any] = {
            "pipeline": pipeline_name,
            "run_id": self.run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "packages": _get_installed_packages(),
                "git_commit": _get_git_commit(),
            },
            "config_snapshot": None,
            "random_seeds": {},
            "inputs": {},
            "stages": [],
            "outputs": {},
            "completed_at_utc": None,
            "status": "running",
        }

    def record_config(self, config: Dict[str, Any]) -> None:
        """Snapshot the full config used for this run."""
        config_json = json.dumps(config, sort_keys=True, default=str)
        self.manifest["config_snapshot"] = config
        self.manifest["config_sha256"] = hashlib.sha256(
            config_json.encode("utf-8")
        ).hexdigest()

    def record_random_seed(self, name: str, seed: int) -> None:
        """Record a random seed used in this run for reproducibility."""
        self.manifest["random_seeds"][name] = seed

    def record_input(self, name: str, file_path: str) -> None:
        """Record an input file with its SHA-256 checksum."""
        path = Path(file_path)
        entry: Dict[str, Any] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
        }
        if path.exists():
            entry["sha256"] = sha256_checksum(file_path)
            entry["size_bytes"] = path.stat().st_size
            entry["modified_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        self.manifest["inputs"][name] = entry

    def record_stage(
        self,
        stage_name: str,
        rows_in: int,
        rows_out: int,
        rows_dropped: int = 0,
        drop_reasons: Optional[Dict[str, int]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Record row-level accounting for a pipeline stage."""
        stage: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_dropped": rows_dropped,
        }
        if drop_reasons:
            stage["drop_reasons"] = drop_reasons
        if notes:
            stage["notes"] = notes
        # Sanity check: rows_in should equal rows_out + rows_dropped
        if rows_in != rows_out + rows_dropped:
            stage["accounting_warning"] = (
                f"rows_in ({rows_in}) != rows_out ({rows_out}) + rows_dropped ({rows_dropped})"
            )
        self.manifest["stages"].append(stage)

    def record_output(self, name: str, file_path: str) -> None:
        """Record an output file with its SHA-256 checksum."""
        path = Path(file_path)
        entry: Dict[str, Any] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
        }
        if path.exists():
            entry["sha256"] = sha256_checksum(file_path)
            entry["size_bytes"] = path.stat().st_size
        self.manifest["outputs"][name] = entry

    def record_parameter(self, key: str, value: Any) -> None:
        """Record an analytical parameter for the manifest."""
        if "parameters" not in self.manifest:
            self.manifest["parameters"] = {}
        self.manifest["parameters"][key] = value

    def finalize(self, status: str = "completed") -> None:
        """Mark the run as completed."""
        self.manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["status"] = status

    def write_manifest(self, output_path: str) -> None:
        """Write the full audit manifest to a JSON file."""
        self.finalize()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest, indent=2, default=str),
            encoding="utf-8",
        )

    def get_summary_lines(self) -> List[str]:
        """Return human-readable summary lines for console output."""
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"Run ID:   {self.run_id}",
            f"Git:      {self.manifest['environment'].get('git_commit', 'n/a')}",
            f"Inputs:   {len(self.manifest['inputs'])} files",
            f"Stages:   {len(self.manifest['stages'])} recorded",
            f"Outputs:  {len(self.manifest['outputs'])} files",
        ]
        total_dropped = sum(
            s.get("rows_dropped", 0) for s in self.manifest["stages"]
        )
        if total_dropped > 0:
            lines.append(f"Total rows dropped across stages: {total_dropped}")
        warnings = [
            s for s in self.manifest["stages"] if "accounting_warning" in s
        ]
        if warnings:
            lines.append(f"Accounting warnings: {len(warnings)} stages")
        return lines
