"""Hermetic test setup: everything (data, MLflow db, models, logs) lives in a temp folder."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TMP = Path(tempfile.mkdtemp(prefix="mlops_test_"))
os.environ["MLOPS_HOME"] = str(TMP)                                   # must be set BEFORE importing src.*
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(TMP / 'mlflow.db').as_posix()}"
CODE_ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    """Run `python -m src.<module> ...` against the temp workspace."""
    return subprocess.run([sys.executable, "-m", *args], cwd=CODE_ROOT, env=os.environ,
                          capture_output=True, text=True)


@pytest.fixture(scope="session", autouse=True)
def trained_workspace():
    (TMP / "data" / "raw").mkdir(parents=True)
    r = run("src.data_gen", "--n", "400", "--out", str(TMP / "data" / "raw" / "disease_data.csv"))
    assert r.returncode == 0, r.stderr
    r = run("src.train")
    assert r.returncode == 0, r.stderr
    return TMP
