"""
Simulated Prometheus-style exporter for prediction/data drift on the
pod-crashloop-classifier model, and the "close the loop" trigger into the
Kubeflow retraining pipeline.

WHAT'S REAL vs WHAT'S SIMULATED (be upfront about this):
  - REAL: the drift statistic itself. Population Stability Index (PSI) is a
    genuine, widely-used technique (common in credit risk / fraud model
    monitoring, not something invented for this POC) for comparing a
    current feature distribution against a baseline. The thresholds used
    below (0.1 / 0.25) are the commonly cited industry rules of thumb, not
    numbers I made up for this demo.
  - REAL: the Prometheus text exposition format this script emits — it's
    valid, scrapeable output. `--serve` starts an actual HTTP server you
    can `curl localhost:9105/metrics` against.
  - SIMULATED: there's no real model serving traffic feeding this. Instead
    of pulling live predictions from KServe, this script generates a
    sequence of synthetic "windows" of feature data, with a deliberate
    distribution shift injected partway through, to demonstrate the drift
    calculation and alerting logic against something that actually drifts.
    A real exporter would instead be a sidecar/job reading recent request
    features + predictions from KServe's request logging (or a feature
    store), not a generator.
  - SIMULATED (not executed): the "trigger Kubeflow retraining" step is a
    stub that prints what it *would* call — a kfp.Client().create_run(...)
    against the pipeline in ../kubeflow/training_pipeline.py — rather than
    an actual network call. There's no live KFP endpoint to call in this
    POC, and I don't want this script to silently no-op on a real API call
    that always fails; making the trigger explicit and printed is more
    honest than half-wiring a call that can never succeed here.
  - NOT BUILT: a real Alertmanager/PrometheusRule wiring, or an Argo Events
    sensor turning a firing alert into a pipeline run. That's the
    production path this stub represents; describing it is part of the
    point of this file (see the docstring on trigger_retraining_pipeline).

Run it:
    python drift_exporter.py            # runs the simulated windows once, prints results
    python drift_exporter.py --serve    # also starts an HTTP server exposing the
                                         # latest window's metrics at /metrics
"""

import argparse
import http.server
import socketserver
import time

import numpy as np

FEATURES = [
    "restart_count",
    "memory_pressure_pct",
    "image_pull_failures",
    "config_change_recency_hrs",
]

# PSI thresholds — standard rules of thumb used in industry model
# monitoring (e.g. credit scoring): below 0.1 is treated as no significant
# shift, 0.1-0.25 as moderate drift worth watching, above 0.25 as
# significant drift that should trigger action.
PSI_WATCH_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

METRICS_PORT = 9105


def generate_window(n_samples: int, seed: int, drift_shift: float = 0.0) -> dict:
    """
    Generates one "window" of feature data, structurally the same
    generator as mlflow_tracking/train_and_register.py's synthetic
    dataset. `drift_shift` pushes memory_pressure_pct upward to simulate
    a real-world scenario like incident_003's node memory pressure drift
    (see ../rag_auto_rca/sample_incidents/incident_003_alert_storm_memory_pressure.md)
    — the same "slow trend crosses a threshold" shape, applied to model
    inputs instead of node memory.
    """
    rng = np.random.default_rng(seed)
    return {
        "restart_count": rng.poisson(lam=2.0, size=n_samples).astype(float),
        "memory_pressure_pct": np.clip(
            rng.uniform(10, 100, size=n_samples) + drift_shift, 0, 100
        ),
        "image_pull_failures": rng.poisson(lam=0.3, size=n_samples).astype(float),
        "config_change_recency_hrs": rng.exponential(scale=24.0, size=n_samples),
    }


def compute_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between a baseline distribution and a
    current one. Bins are defined by baseline quantiles so each baseline
    bucket has ~equal weight; PSI measures how much the current window's
    bucket proportions have shifted away from that.
    """
    quantile_edges = np.quantile(baseline, np.linspace(0, 1, buckets + 1))
    quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf  # catch outliers on either tail

    baseline_counts, _ = np.histogram(baseline, bins=quantile_edges)
    current_counts, _ = np.histogram(current, bins=quantile_edges)

    baseline_pct = np.clip(baseline_counts / baseline_counts.sum(), 1e-6, None)
    current_pct = np.clip(current_counts / current_counts.sum(), 1e-6, None)

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def compute_drift_report(baseline: dict, current: dict) -> dict:
    """Returns per-feature PSI plus an aggregate (max) drift score for the window."""
    per_feature = {feature: compute_psi(baseline[feature], current[feature]) for feature in FEATURES}
    aggregate = max(per_feature.values())
    return {"per_feature_psi": per_feature, "aggregate_psi": aggregate}


def trigger_retraining_pipeline(window_index: int, report: dict):
    """
    This is the "self-healing" closing-the-loop step: when drift crosses
    the alert threshold, kick off the Kubeflow training pipeline defined in
    ../kubeflow/training_pipeline.py so a fresh model gets trained against
    recent data instead of waiting for a human to notice degraded
    predictions.

    NOT EXECUTED — this only prints the call that a real implementation
    would make. In production this function's body would be roughly:

        from kfp.client import Client
        client = Client(host="http://kfp-pipelines.mlops-poc.svc.cluster.local")
        client.create_run_from_pipeline_package(
            "../kubeflow/training_pipeline.yaml",
            arguments={"seed": window_index},
        )

    and in a fuller production setup, this whole script wouldn't be the
    thing deciding to call that — it would be a PrometheusRule alert
    firing on this exporter's `crashloop_model_drift_psi` metric, routed
    through Alertmanager to an Argo Events webhook sensor, which then
    triggers the KFP run. That indirection matters: it means the
    retraining trigger is driven by the same alerting stack already
    watching everything else, not a bespoke one-off integration.
    """
    print(
        f"  [TRIGGER] window {window_index}: aggregate PSI "
        f"{report['aggregate_psi']:.3f} > {PSI_ALERT_THRESHOLD} — "
        f"would call kfp.Client().create_run_from_pipeline_package("
        f"'../kubeflow/training_pipeline.yaml', ...) here. Not executed "
        f"(no live KFP endpoint in this POC)."
    )


def format_prometheus_metrics(report: dict, window_index: int) -> str:
    """Renders the latest drift report as valid Prometheus text exposition format."""
    lines = [
        "# HELP crashloop_model_drift_psi Population Stability Index vs training baseline",
        "# TYPE crashloop_model_drift_psi gauge",
    ]
    for feature, psi in report["per_feature_psi"].items():
        lines.append(f'crashloop_model_drift_psi{{feature="{feature}",window="{window_index}"}} {psi:.6f}')

    lines += [
        "# HELP crashloop_model_drift_psi_aggregate Max per-feature PSI for the window",
        "# TYPE crashloop_model_drift_psi_aggregate gauge",
        f"crashloop_model_drift_psi_aggregate{{window=\"{window_index}\"}} {report['aggregate_psi']:.6f}",
        "# HELP crashloop_model_drift_alert 1 if aggregate PSI crossed the alert threshold",
        "# TYPE crashloop_model_drift_alert gauge",
        f"crashloop_model_drift_alert{{window=\"{window_index}\"}} "
        f"{1 if report['aggregate_psi'] > PSI_ALERT_THRESHOLD else 0}",
    ]
    return "\n".join(lines) + "\n"


def run_simulation(n_windows: int = 6, n_samples: int = 500) -> str:
    """
    Runs a sequence of windows: the first few are stable (drift_shift=0),
    then a gradual upward shift in memory_pressure_pct is injected,
    mirroring a real degradation pattern rather than a single-step jump.
    Returns the last window's Prometheus-formatted metrics text (also used
    by --serve).
    """
    baseline = generate_window(n_samples=2000, seed=0, drift_shift=0.0)

    latest_metrics_text = ""
    print(f"Baseline built from {n_samples * 4} synthetic samples (seed=0, no drift).\n")

    for window_index in range(n_windows):
        # Windows 0-1 stable; drift ramps up from window 2 onward. Kept
        # small relative to the feature's 10-100 range so PSI lands in a
        # realistic band (roughly 0.1-1.0) instead of the degenerate,
        # everything-piles-up-at-the-clip-boundary values a larger shift
        # would produce.
        drift_shift = 0.0 if window_index < 2 else (window_index - 1) * 3.0
        current = generate_window(n_samples=n_samples, seed=100 + window_index, drift_shift=drift_shift)

        report = compute_drift_report(baseline, current)
        status = (
            "ALERT" if report["aggregate_psi"] > PSI_ALERT_THRESHOLD
            else "WATCH" if report["aggregate_psi"] > PSI_WATCH_THRESHOLD
            else "OK"
        )

        print(f"Window {window_index} [{status}] aggregate PSI={report['aggregate_psi']:.3f}")
        for feature, psi in report["per_feature_psi"].items():
            print(f"    {feature:<28} PSI={psi:.3f}")

        if report["aggregate_psi"] > PSI_ALERT_THRESHOLD:
            trigger_retraining_pipeline(window_index, report)

        latest_metrics_text = format_prometheus_metrics(report, window_index)
        print()

    return latest_metrics_text


def serve_metrics(metrics_text: str, port: int = METRICS_PORT):
    """Starts a minimal HTTP server exposing the given metrics text at /metrics."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - required method name for BaseHTTPRequestHandler
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            body = metrics_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # quiet down default request logging for a demo script

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Serving latest window's metrics at http://localhost:{port}/metrics (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve", action="store_true", help="After running the simulation, serve /metrics over HTTP"
    )
    parser.add_argument("--port", type=int, default=METRICS_PORT)
    args = parser.parse_args()

    latest_metrics_text = run_simulation()

    if args.serve:
        serve_metrics(latest_metrics_text, port=args.port)
    else:
        print("--serve not passed; skipping HTTP server. Latest window's Prometheus output:\n")
        print(latest_metrics_text)


if __name__ == "__main__":
    main()
