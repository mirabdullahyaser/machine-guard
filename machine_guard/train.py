"""Train a Machine Guard model, log it to MLflow and, if asked, register it.

Run from the repo root:
    python -m machine_guard.train                                   # the Day 1 champion settings
    python -m machine_guard.train --model decision_tree --class-weight none --no-engineered
    python -m machine_guard.train --register --alias champion       # train, register, point the alias at it
    python -m machine_guard.train --help                            # every option

Every run logs the same things, so runs can be compared: params (model, class_weight, engineered,
seed and the model settings), metrics (recall, precision, f1, accuracy, missed_failures,
false_alarms, cost) and tags (git_commit, data_md5).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # keep the terminal output short
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")  # harmless MLflow hint
import mlflow  # noqa: E402
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from machine_guard.features import REGISTERED_MODEL_NAME, SEED, build_pipeline, load_data, split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "data/raw/ai4i2020.csv"
EXPERIMENT = "machine-guard"
THRESHOLD = 0.5            # probability above which we call a failure (fixed for the course)
COST_MISSED = 20           # one missed breakdown costs 20 times ...
COST_FALSE_ALARM = 1       # ... one false alarm (the Day 1 ratio)


# ------------------------------------------------------------------ MLflow set-up (given)
def setup_mlflow(experiment: str = EXPERIMENT) -> str:
    """Point MLflow at the project database and pick the experiment. Works from any folder."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return uri


# ------------------------------------------------------------------ lineage helpers (given)
def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def data_md5(path: Path) -> str:
    """The data version: the md5 that DVC recorded in `<file>.dvc`, or the file's own md5 before DVC."""
    dvc_file = Path(str(path) + ".dvc")
    if dvc_file.exists():
        return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    return file_md5(path)


def git_commit() -> tuple[str, bool]:
    """The current commit and whether there are uncommitted changes (then the commit alone can't rebuild the run)."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=REPO_ROOT, text=True).strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


# ------------------------------------------------------------------ Lab 1: train and score
def evaluate(y_true, y_pred) -> dict:
    """Shop-floor scorecard for one model on the test set."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "missed_failures": int(fn),
        "false_alarms": int(fp),
        "cost": int(fn * COST_MISSED + fp * COST_FALSE_ALARM),
    }


def model_settings(args) -> dict:
    """Only the settings the user actually gave; build_pipeline fills in the course defaults."""
    settings = {"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
                "max_depth": args.max_depth}
    return {k: v for k, v in settings.items() if v is not None}


def train(args):
    data_path = REPO_ROOT / args.data
    df = load_data(str(data_path))
    X_train, X_test, y_train, y_test = split(df)

    class_weight = None if args.class_weight == "none" else args.class_weight
    pipe = build_pipeline(args.model, class_weight, args.engineered, **model_settings(args))
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    metrics = evaluate(y_test, pred)
    return pipe, metrics, X_test, y_test, proba, data_path, len(df)


# ------------------------------------------------------------------ Lab 2: track and register
def log_run(args, pipe, metrics, X_test, proba, data_path, n_rows) -> tuple[str, str | None]:
    commit, dirty = git_commit()
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "model": args.model,
            "class_weight": args.class_weight,
            "engineered": args.engineered,
            "seed": SEED,
            "threshold": THRESHOLD,
            # every setting the estimator really used, defaults included, so nothing is left to memory
            **{k: v for k, v in pipe.named_steps["model"].get_params().items()
               if k in ("n_estimators", "min_samples_leaf", "max_depth", "max_iter")},
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": commit,
            "git_dirty": dirty,
            "data_md5": data_md5(data_path),
            "data_path": args.data,
            "data_rows": n_rows,
        })
        signature = infer_signature(X_test, proba)
        mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature,
                                 input_example=X_test.head(3))
        version = None
        if args.register:
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL_NAME)
            version = mv.version
            client = MlflowClient()
            for key in ("git_commit", "data_md5"):  # copy lineage onto the model version itself
                client.set_model_version_tag(REGISTERED_MODEL_NAME, version, key,
                                             commit if key == "git_commit" else data_md5(data_path))
            if args.alias:
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, version)
        return run.info.run_id, version


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a Machine Guard model and log it to MLflow.")
    p.add_argument("--model", default="random_forest",
                   choices=["random_forest", "decision_tree", "logistic_regression"])
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "balanced_subsample", "none"],
                   help="balanced_subsample works for random_forest only")
    p.add_argument("--engineered", default=True, action=argparse.BooleanOptionalAction,
                   help="add power_w and temp_diff_k (use --no-engineered to switch off)")
    p.add_argument("--n-estimators", type=int, help="random forest: number of trees (default 200)")
    p.add_argument("--min-samples-leaf", type=int, help="random forest: smallest leaf (default 5)")
    p.add_argument("--max-depth", type=int, help="tree depth limit (decision tree default 6)")
    p.add_argument("--data", default=DEFAULT_DATA, help="CSV path relative to the repo root")
    p.add_argument("--run-name", help="a readable name for the run in the MLflow UI")
    p.add_argument("--register", action="store_true", help=f"register the model as '{REGISTERED_MODEL_NAME}'")
    p.add_argument("--alias", help="with --register: point this alias (champion, challenger) at the new version")
    p.add_argument("--no-mlflow", action="store_true", help="train and print only, log nothing")
    args = p.parse_args(argv)
    if args.alias and not args.register:
        p.error("--alias needs --register")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    pipe, metrics, X_test, y_test, proba, data_path, n_rows = train(args)

    print(f"Data: {args.data} ({n_rows:,} rows), test set {len(X_test):,} rows")
    print(f"Model: {args.model}, class_weight={args.class_weight}, engineered={args.engineered} "
          f"{model_settings(args) or ''}".rstrip())
    print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()))
    failures = int(y_test.sum())
    print(f"Caught {failures - metrics['missed_failures']} of {failures} failures, "
          f"{metrics['false_alarms']} false alarms")

    if args.no_mlflow:
        return
    uri = setup_mlflow()
    run_id, version = log_run(args, pipe, metrics, X_test, proba, data_path, n_rows)
    print(f"Logged run {run_id} to experiment '{EXPERIMENT}' ({uri})")
    if version:
        alias = f", alias '{args.alias}'" if args.alias else ""
        print(f"Registered '{REGISTERED_MODEL_NAME}' version {version}{alias}")


if __name__ == "__main__":
    main()
