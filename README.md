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
```

## Live deployment status

- **`mlflow_tracking/train_and_register.py`** — beyond local testing, this
  now also runs as a real Kubernetes `Job` on a live EKS cluster
  (`gpu-poc-cluster`, reused from `eks-gpu-poc`). Built via Kaniko running
  in-cluster (no local Docker/CloudShell available), pushed to ECR, and
  confirmed completing with the same promotion-gate output as the local
  runs. See `mlflow_tracking/Dockerfile`, `kaniko-build-job.yaml`, and
  `k8s-job.yaml`.
- **`rag_auto_rca/`** — retrieval upgraded from TF-IDF to real sentence
  embeddings (`sentence-transformers/all-MiniLM-L6-v2`, local, no API key)
  stored and queried in a real Pinecone index. Verified end-to-end against
  live Pinecone with all 3 sample incidents retrieving correctly, real
  cosine-similarity scores in the 0.62-0.75 range for correct matches vs.
  0.07-0.44 for non-matches — a clearer separation than TF-IDF's closer
  0.33-0.45 range on the same 3 queries. Generation step (Anthropic API
  call / template fallback) unchanged — only retrieval was swapped.
- Everything else in this repo is still at the status described inline in
  each component's own comments (some fully tested locally, some
  designed-but-not-deployed) — not yet re-verified against a live cluster.
