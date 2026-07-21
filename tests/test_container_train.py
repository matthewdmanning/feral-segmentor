"""Contracts for the cloud-training container entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_TRAIN = REPOSITORY_ROOT / "scripts" / "container_train.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_container_training_uses_configured_persistent_mlflow_tracking_uri(tmp_path):
    """Training reports to the shared MLflow server, not a local SQLite store."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "gcloud.log"
    data_dir = tmp_path / "data"

    _write_executable(
        bin_dir / "gcloud",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${CALL_LOG}\"\n",
    )
    python_log = tmp_path / "python.log"
    _write_executable(
        bin_dir / "python",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${PYTHON_LOG}\"\n",
    )
    _write_executable(
        bin_dir / "mlflow",
        "#!/usr/bin/env bash\necho 'local MLflow must not start' >&2\nexit 99\n",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "PYTHON_LOG": str(python_log),
        "DATA_DIR": str(data_dir),
        "GCS_BUCKET": "feral-training",
        "GCS_DATA_PREFIX": "inputs/baseline",
        "MLFLOW_TRACKING_URI": "https://mlflow.example.run.app",
    }

    subprocess.run(
        ["bash", str(CONTAINER_TRAIN)],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text().splitlines()
    assert all("/mlflow/" not in call for call in calls)
    assert (
        f"-m feral_vision.training.trainer --config-name runs/baseline "
        f"data.root={data_dir} tracking.tracking_uri=https://mlflow.example.run.app"
    ) in python_log.read_text().splitlines()
