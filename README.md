## Architecture / flow

```
                    ┌─────────────────────────┐
                    │   kubeflow/              │
                    │   training_pipeline.py   │
                    │   (KFP DAG, compiled     │
                    │   to YAML — not run on   │
   drift alert  ───▶│   a live cluster)        │
   triggers this    │                          │
                    │  fetch data -> train ->  │
                    │  evaluate -> [if F1 ok]  │
                    │  register + promote      │
                    └────────────┬─────────────┘
                                 │ logs experiments, registers
                                 │ model versions, sets
                                 │ "production" alias if it wins
                                 ▼
                    ┌─────────────────────────┐
                    │   mlflow_tracking/       │
                    │   train_and_register.py │
                    │                          │
                    │  Model Registry:         │
                    │  pod-crashloop-classifier│
                    │  @production alias       │
                    └────────────┬─────────────┘
                                 │ (manual/CI artifact sync —
                                 │  see kserve/inferenceservice.yaml
                                 │  comments for why this isn't automatic)
                                 ▼
                    ┌─────────────────────────┐
                    │   kserve/                │
                    │   inferenceservice.yaml  │
                    │                          │
                    │  Serves @production as a │
                    │  REST endpoint. Canary   │
                    │  traffic-split supported │
                    │  for new versions.       │
                    └────────────┬─────────────┘
                                 │ (in a real system: live prediction
                                 │  traffic/features flow out to
                                 │  monitoring)
                                 ▼
                    ┌─────────────────────────┐
                    │   monitoring/            │
                    │   drift_exporter.py      │
                    │                          │
                    │  PSI drift score vs.     │
                    │  training baseline,      │
                    │  Prometheus /metrics     │
                    └────────────┬─────────────┘
                                 │ drift crosses alert threshold
                                 └──────── triggers a new Kubeflow run ────┐
                                                                            │
                                          (loop closes back to the top) ◀──┘
``
