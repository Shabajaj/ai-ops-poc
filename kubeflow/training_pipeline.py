"""
Kubeflow Pipelines (KFP SDK v2) DAG wrapping the MLflow training workflow
from ../mlflow_tracking/train_and_register.py as a pipeline:

    fetch/prepare data -> train -> evaluate -> [conditionally] register + promote

WHAT'S REAL vs WHAT'S CONCEPTUAL (important to be upfront about):
  - REAL: this file compiles with the actual `kfp` SDK into a valid
    Argo-Workflows-backed pipeline YAML spec (see `if __name__ == "__main__"`
    at the bottom). That compile step is tested and runnable locally with
    no cluster required — `pip install kfp` and run this file.
  - CONCEPTUAL / NOT RUN: I don't have a live Kubeflow Pipelines cluster for
    this interview timeline, so this pipeline has never actually been
    *submitted* to a KFP endpoint or executed end-to-end. Everything below
    about "what happens when this runs" describes the intended behavior of
    a valid spec, not an observed result.
  - Each `@dsl.component` becomes its own container at runtime (KFP builds
    a lightweight container from `base_image` + `packages_to_install` for
    each one). On a real cluster this means each pipeline step runs in
    its own pod — that's real KFP behavior, not simplified for the demo.
  - The training logic here (make_synthetic_dataset, RandomForest, F1)
    intentionally mirrors train_and_register.py rather than importing it
    directly, because KFP lightweight components must be self-contained
    (they ship as source to a fresh container, not as a reference into this
    repo's file tree). A production version would instead package the
    training code into a proper container image pushed to a registry, and
    have each component reference that image directly — that avoids
    duplicating logic between the standalone script and the pipeline, which
    is a real maintenance cost of the approach shown here.
  - The register_and_promote component assumes a reachable MLflow tracking
    server (MLFLOW_TRACKING_URI pointed at a real server, not the local
    sqlite file train_and_register.py uses standalone) — that's the one
    piece of real infrastructure this pipeline depends on beyond KFP itself.
"""

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

BASE_IMAGE = "python:3.11-slim"
TRAIN_PACKAGES = ["scikit-learn==1.5.0", "pandas==2.2.2", "joblib==1.4.2"]
MLFLOW_PACKAGES = ["mlflow==2.14.1"]

# Same promotion-gate concept as train_and_register.py: a candidate model
# must beat this bar to even be considered for the registry/promotion step.
# In train_and_register.py the gate is "beat current production F1"; this
# adds a floor check upstream of that, since a pipeline run might produce a
# model too weak to be worth registering at all, gate or no gate.
MIN_ACCEPTABLE_F1 = 0.55


@dsl.component(base_image=BASE_IMAGE, packages_to_install=TRAIN_PACKAGES)
def prepare_data(n_samples: int, noise: float, seed: int, output_data: Output[Dataset]):
    """
    Fetch/prepare step. Generates the same synthetic feature set as
    train_and_register.py's make_synthetic_dataset (restart_count,
    memory_pressure_pct, image_pull_failures, config_change_recency_hrs).

    In a real pipeline this component would instead query Prometheus/
    kube-state-metrics history and an incident tracker for labels — kept
    synthetic here for the same reason train_and_register.py is synthetic:
    this POC demonstrates the MLOps workflow, not a validated predictor.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    restart_count = rng.poisson(lam=2.0, size=n_samples).astype(float)
    memory_pressure_pct = rng.uniform(10, 100, size=n_samples)
    image_pull_failures = rng.poisson(lam=0.3, size=n_samples).astype(float)
    config_change_recency_hrs = rng.exponential(scale=24.0, size=n_samples)

    risk_score = (
        0.35 * (restart_count / (restart_count.max() + 1e-6))
        + 0.30 * (memory_pressure_pct / 100.0)
        + 0.20 * (image_pull_failures / (image_pull_failures.max() + 1e-6))
        + 0.15 * (1.0 / (1.0 + config_change_recency_hrs / 6.0))
    )
    risk_score += rng.normal(0, noise, size=n_samples)
    crashloop_risk = (risk_score > np.median(risk_score)).astype(int)

    df = pd.DataFrame(
        {
            "restart_count": restart_count,
            "memory_pressure_pct": memory_pressure_pct,
            "image_pull_failures": image_pull_failures,
            "config_change_recency_hrs": config_change_recency_hrs,
            "crashloop_risk": crashloop_risk,
        }
    )
    df.to_csv(output_data.path, index=False)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=TRAIN_PACKAGES)
def train_model(input_data: Input[Dataset], seed: int, output_model: Output[Model]):
    """Trains the RandomForest classifier and writes it out as a KFP Model artifact."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    feature_cols = [
        "restart_count",
        "memory_pressure_pct",
        "image_pull_failures",
        "config_change_recency_hrs",
    ]
    X = df[feature_cols].values
    y = df["crashloop_risk"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "X_test": X_test, "y_test": y_test}, output_model.path)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=TRAIN_PACKAGES)
def evaluate_model(input_model: Input[Model], output_metrics: Output[Metrics]) -> float:
    """Scores the held-out test split and logs F1 as a KFP Metrics artifact."""
    import joblib
    from sklearn.metrics import f1_score

    bundle = joblib.load(input_model.path)
    preds = bundle["model"].predict(bundle["X_test"])
    f1 = float(f1_score(bundle["y_test"], preds))

    output_metrics.log_metric("f1_score", f1)
    return f1


@dsl.component(base_image=BASE_IMAGE, packages_to_install=MLFLOW_PACKAGES + TRAIN_PACKAGES)
def register_and_promote(
    input_model: Input[Model],
    f1_score_value: float,
    mlflow_tracking_uri: str,
    model_name: str,
):
    """
    Registers the trained model in the MLflow Model Registry and applies the
    same promotion gate as train_and_register.py: only move the "production"
    alias if this run beats the current production model's F1.

    NOTE: this component needs a real, reachable MLflow tracking server —
    unlike train_and_register.py's standalone default of a local sqlite
    file, a pipeline step running in its own pod can't share a local file
    with the MLflow client running elsewhere. `mlflow_tracking_uri` would
    point at the docker-compose server in ../mlflow_tracking/docker-compose.yml
    (or a real hosted MLflow instance) in an actual deployment.
    """
    import joblib
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    bundle = joblib.load(input_model.path)
    model = bundle["model"]

    with mlflow.start_run() as run:
        mlflow.log_metric("f1_score", f1_score_value)
        mlflow.sklearn.log_model(model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"

    result = mlflow.register_model(model_uri, model_name)
    new_version = result.version

    try:
        current = client.get_model_version_by_alias(model_name, "production")
        current_run = client.get_run(current.run_id)
        current_f1 = current_run.data.metrics.get("f1_score")
    except mlflow.exceptions.MlflowException:
        current_f1 = None

    if current_f1 is None or f1_score_value > current_f1:
        client.set_registered_model_alias(model_name, "production", new_version)
        print(f"[PROMOTE] version {new_version}, F1={f1_score_value:.4f} (prior={current_f1})")
    else:
        print(f"[REJECT] version {new_version}, F1={f1_score_value:.4f} did not beat {current_f1:.4f}")


@dsl.pipeline(
    name="pod-crashloop-classifier-training",
    description=(
        "Fetch data -> train -> evaluate -> conditionally register+promote "
        "the pod CrashLoopBackOff risk classifier."
    ),
)
def training_pipeline(
    n_samples: int = 2000,
    noise: float = 0.25,
    seed: int = 42,
    mlflow_tracking_uri: str = "http://mlflow-server.mlops-poc.svc.cluster.local:5000",
    model_name: str = "pod-crashloop-classifier",
):
    prepare_task = prepare_data(n_samples=n_samples, noise=noise, seed=seed)

    train_task = train_model(input_data=prepare_task.outputs["output_data"], seed=seed)

    evaluate_task = evaluate_model(input_model=train_task.outputs["output_model"])

    # Conditional branch: only run the register/promote step if the model
    # clears the minimum bar. This is the pipeline-level gate; the
    # beat-current-production check happens inside register_and_promote
    # itself, mirroring train_and_register.py's two-layer logic (is this
    # model good enough to register at all, then is it good enough to
    # actually serve traffic).
    # evaluate_model has two outputs (the Metrics artifact + the returned
    # float), so KFP requires referencing the return value by its implicit
    # name "Output" rather than the single-output shorthand `.output`.
    with dsl.If(evaluate_task.outputs["Output"] >= MIN_ACCEPTABLE_F1, name="clears-minimum-f1-bar"):
        register_and_promote(
            input_model=train_task.outputs["output_model"],
            f1_score_value=evaluate_task.outputs["Output"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
        )


def compile_pipeline(output_path: str = "training_pipeline.yaml"):
    """
    Compiles this pipeline to an Argo-Workflows-backed YAML spec. This is
    the part of this file that's genuinely tested/runnable without a
    cluster — `python training_pipeline.py` produces a real spec you can
    open and read, or later upload to a KFP UI / submit via `kfp run`.
    """
    compiler.Compiler().compile(pipeline_func=training_pipeline, package_path=output_path)
    print(f"Compiled pipeline spec written to {output_path}")


if __name__ == "__main__":
    compile_pipeline()
