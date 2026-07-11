"""
MLflow training + registration for a pod CrashLoopBackOff risk classifier.

SCOPE (read this first): the training data is synthetic — generated in this
script, not pulled from a real cluster. The point of this POC is to show the
MLflow workflow (experiment tracking -> Model Registry -> promotion gate),
not to claim a validated CrashLoopBackOff predictor. A real version would
train on features pulled from Prometheus/kube-state-metrics history and
actual crash labels from an incident tracker.

Features (all synthetic, roughly correlated with real signals SREs watch):
  - restart_count             : container restart count in the observation window
  - memory_pressure_pct       : % of memory limit the pod was sitting at before restart
  - image_pull_failures       : count of ImagePullBackOff/ErrImagePull events
  - config_change_recency_hrs : hours since the last ConfigMap/Secret/image change
Label:
  - crashloop_risk (0/1)      : whether the pod actually entered CrashLoopBackOff

Run this multiple times to see the promotion gate in action:
  python train_and_register.py                # run 1: nothing to beat -> auto-promotes
  python train_and_register.py --weak          # run 2: deliberately bad model -> rejected
  python train_and_register.py --strong        # run 3: better model -> promoted
"""

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "pod-crashloop-risk"
MODEL_NAME = "pod-crashloop-classifier"
PRODUCTION_ALIAS = "production"

# Model Registry requires a database-backed backend store — the default
# plain file store (mlruns/) raises "Registry functionality is unavailable"
# on current MLflow versions. sqlite:// is the simplest DB-backed option
# for a single-machine POC; a real deployment would point this at a
# managed Postgres instance (see docker-compose.yml for that variant).
#
# Overridable via MLFLOW_TRACKING_URI so the same container image can run
# standalone (local sqlite file) or in a cluster pointed at a shared,
# persistent MLflow server — see Dockerfile for why that distinction
# matters when this runs as a Kubernetes Job.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def make_synthetic_dataset(n_samples: int, noise: float, seed: int):
    """
    Generates a synthetic, roughly-plausible dataset for the crashloop
    classifier. `noise` controls how cleanly the label follows the
    features — used to simulate a "weak" vs "strong" training run without
    needing two separate real datasets.
    """
    rng = np.random.default_rng(seed)

    restart_count = rng.poisson(lam=2.0, size=n_samples).astype(float)
    memory_pressure_pct = rng.uniform(10, 100, size=n_samples)
    image_pull_failures = rng.poisson(lam=0.3, size=n_samples).astype(float)
    config_change_recency_hrs = rng.exponential(scale=24.0, size=n_samples)

    # Ground-truth signal: risk rises with restarts, memory pressure, pull
    # failures, and recent config changes. This is a hand-built heuristic,
    # not a real physical model — good enough to make the classifier's
    # job "learnable" for a demo.
    risk_score = (
        0.35 * (restart_count / (restart_count.max() + 1e-6))
        + 0.30 * (memory_pressure_pct / 100.0)
        + 0.20 * (image_pull_failures / (image_pull_failures.max() + 1e-6))
        + 0.15 * (1.0 / (1.0 + config_change_recency_hrs / 6.0))
    )
    risk_score += rng.normal(0, noise, size=n_samples)
    crashloop_risk = (risk_score > np.median(risk_score)).astype(int)

    X = np.column_stack(
        [restart_count, memory_pressure_pct, image_pull_failures, config_change_recency_hrs]
    )
    y = crashloop_risk
    return X, y


def get_current_production_f1(client: MlflowClient) -> float | None:
    """Returns the F1 score of the current 'production' alias, or None if unset."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None  # no registered model / no production alias yet

    run = client.get_run(version.run_id)
    return run.data.metrics.get("f1_score")


def train_and_log(noise: float, seed: int) -> tuple[str, float]:
    """Trains one model, logs it to MLflow, returns (run_id, f1_score)."""
    X, y = make_synthetic_dataset(n_samples=2000, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 50 if noise > 0.3 else 200,  # deliberately undersized for --weak
            "max_depth": 3 if noise > 0.3 else 8,
            "random_state": seed,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_param("noise", noise)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test[:1],
        )

        print(f"Run {run.info.run_id}: F1 = {f1:.4f} (params={params})")
        return run.info.run_id, f1


def register_and_maybe_promote(client: MlflowClient, run_id: str, new_f1: float):
    """
    Registers the run's model as a new version, then applies the promotion
    gate: only move the 'production' alias to this version if it beats the
    current production model's F1. If no production model exists yet, the
    first version is promoted automatically (nothing to compare against).
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    new_version = result.version

    current_f1 = get_current_production_f1(client)

    if current_f1 is None:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, new_version)
        print(
            f"[PROMOTE] No existing production model — version {new_version} "
            f"(F1={new_f1:.4f}) promoted by default."
        )
        return

    if new_f1 > current_f1:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, new_version)
        print(
            f"[PROMOTE] Version {new_version} (F1={new_f1:.4f}) beat current "
            f"production (F1={current_f1:.4f}) — alias moved."
        )
    else:
        print(
            f"[REJECT] Version {new_version} (F1={new_f1:.4f}) did not beat "
            f"current production (F1={current_f1:.4f}) — alias unchanged. "
            f"Registered but not serving."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weak", action="store_true", help="Train a deliberately weak model (high noise, small forest)"
    )
    parser.add_argument(
        "--strong", action="store_true", help="Train a strong model (low noise, larger forest)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed")
    args = parser.parse_args()

    if args.weak and args.strong:
        print("Pass only one of --weak / --strong", file=sys.stderr)
        sys.exit(1)

    noise = 0.6 if args.weak else (0.05 if args.strong else 0.25)
    seed = args.seed if args.seed is not None else (1 if args.weak else (3 if args.strong else 42))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    run_id, f1 = train_and_log(noise=noise, seed=seed)
    register_and_maybe_promote(client, run_id, f1)


if __name__ == "__main__":
    main()
