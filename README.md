# AI Ops POC — MLflow, Kubeflow, KServe, RAG Auto-RCA

## What this is
A same-day, interview-scoped POC demonstrating the MLOps/AIOps toolchain
named in a "Kubernetes Platform Engineer — AI Infrastructure" job description:
MLflow, Kubeflow Pipelines, KServe, and a RAG-based auto-RCA assistant. It's
meant to sit alongside [`eks-gpu-poc`](https://github.com/Shabajaj/eks-gpu-poc)
(Terraform/EKS GPU node provisioning) as a second, complementary artifact —
that one shows infra provisioning for GPU workloads, this one shows the
ML lifecycle that would run on top of that infra.

**Read this before the interview:** I'm a DevOps/SRE engineer (4 years,
AKS/OpenShift/Terraform/CI-CD, Dynatrace/Prometheus observability), not a
formal software or ML engineer. This POC demonstrates that I can read the
MLOps toolchain's architecture, wire the pieces together correctly, and
reason about the production gaps — not that I have production ML engineering
experience. Every component below is scoped honestly: what's real/tested vs.
what's a well-commented conceptual sketch is called out explicitly, in the
code and here.

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

A fifth component, `rag_auto_rca/`, is deliberately drawn separately — it's
not part of the training/serving/monitoring loop above. It answers a
different question ("given a new alert, what does it look like and what fixed
it last time"), pulling from a small corpus of past incident postmortems
rather than from this pipeline's own model. It's included because
auto-RCA/RAG is explicitly named in the JD, not because it's architecturally
downstream of the other four pieces.

## Components

| Path | What it is | Tested/runs locally? |
|---|---|---|
| `mlflow_tracking/train_and_register.py` | Trains a RandomForest CrashLoopBackOff-risk classifier, logs to MLflow, registers + promotion-gates versions | Yes — 3 runs (auto-promote, rejected, promoted) |
| `mlflow_tracking/docker-compose.yml` | Optional MLflow server+UI for a visual registry demo | Optional, not required |
| `rag_auto_rca/` | TF-IDF retrieval over 3 incident postmortems + RCA synthesis (real Anthropic API call or template fallback) | Yes, in fallback mode |
| `kserve/inferenceservice.yaml` | KServe manifest serving the registered model, with canary traffic-split | Valid manifest; not deployed to a live cluster |
| `kubeflow/training_pipeline.py` | KFP SDK pipeline wrapping the training step as a DAG, compiles to YAML | Compile step runs locally; not submitted to a live KFP cluster |
| `monitoring/drift_exporter.py` | Simulated Prometheus exporter: PSI drift score vs. baseline, stub trigger into the Kubeflow pipeline | Yes — runs end-to-end, `--serve` exposes real `/metrics` |

## Running what's runnable

```bash
cd mlflow_tracking
pip install mlflow scikit-learn numpy
python train_and_register.py            # run 1: auto-promotes (nothing to beat)
python train_and_register.py --weak     # run 2: correctly rejected
python train_and_register.py --strong   # run 3: promoted

cd ../rag_auto_rca
pip install scikit-learn
python query_rca.py "pods crashing right after we rebuilt the base image, no error logs"
# add ANTHROPIC_API_KEY to your environment first to use the real LLM path
# instead of the template fallback

cd ../monitoring
python drift_exporter.py                # prints 6 simulated windows, PSI per feature,
                                          # fires the (stubbed, unexecuted) retrain trigger
python drift_exporter.py --serve         # also serves the last window at
                                          # http://localhost:9105/metrics

cd ../kubeflow
pip install kfp
python training_pipeline.py              # compiles training_pipeline.yaml
```

`kserve/inferenceservice.yaml` isn't something you "run" without a real
KServe/Knative cluster — walk through it as a manifest.

## Known limitations / production differences (be upfront if asked)

- **Synthetic training data everywhere.** The classifier's features
  (restart count, memory pressure %, image pull failures, config-change
  recency) are generated, not pulled from real cluster history. A real
  version would source these from Prometheus/kube-state-metrics and label
  them from an incident tracker. This POC is about the *workflow*
  (train → track → register → promote → serve → monitor → retrain), not a
  validated predictor.
- **TF-IDF, not real embeddings, for RCA retrieval.** Deliberate same-day
  scope trade-off — TF-IDF needs no external dependencies and is
  deterministic, but only catches exact-token overlap, not semantic
  similarity. At 3 documents this gap is invisible; at a real postmortem
  corpus it would matter. A production version would use sentence
  embeddings in FAISS/Pinecone/pgvector. See the comment in `ingest.py`.
- **No live cluster for KServe or Kubeflow.** Both `inferenceservice.yaml`
  and `training_pipeline.py` are real, valid, complete artifacts — the
  YAML would apply, the pipeline compiles to a real Argo Workflows spec —
  but neither has actually been deployed/run end-to-end against a running
  cluster in this POC. I'm explicit about this distinction throughout the
  code comments: "would work" is not the same claim as "I watched it work."
- **MLflow Registry → KServe artifact sync is a manual gap.** KServe reads
  model artifacts from object storage via `storageUri`, not from the
  MLflow Registry API directly. A real pipeline needs a sync step
  (CI job or script) copying a newly promoted version's artifacts to the
  path KServe serves from — not built here, called out explicitly in
  `kserve/inferenceservice.yaml`.
- **Drift exporter simulates its own drifting data**, rather than reading
  real request/prediction logs from KServe. The PSI statistic and
  Prometheus output format are real; the input feed is a generator with an
  injected shift, not live traffic.
- **Retraining trigger is a stub.** `drift_exporter.py` prints the
  `kfp.Client().create_run_from_pipeline_package(...)` call it would make
  rather than making it, and there's no real Alertmanager/Argo Events
  wiring turning a firing alert into that call. Both are described in
  comments as the production path this stands in for.
- **Kubeflow pipeline duplicates training logic** rather than importing
  `train_and_register.py` directly, because KFP lightweight components
  ship as self-contained source to their own container. A real version
  would package the training code into a proper image referenced by each
  component, rather than maintaining the logic in two places.

## How to explain this in an interview

- **MLflow**: "The training script has a real promotion gate — a new model
  version only gets the `production` alias if it beats the current
  production model's F1 score on held-out data. I tested all three paths:
  first run auto-promotes since there's nothing to beat, a deliberately
  undersized/noisy model gets correctly rejected, and a stronger model gets
  promoted. That gate is what keeps the registry from just serving
  whatever was trained most recently."
- **Kubeflow**: "This compiles to a real Argo-Workflows-backed pipeline
  spec with the KFP SDK — fetch data, train, evaluate, and a conditional
  branch that only registers/promotes if the model clears a minimum F1
  bar. I haven't run it against a live KFP cluster for this timeline, but
  the DAG structure and the conditional gating logic are the same shape
  used in the drift-triggered retraining loop I'd want in production."
- **KServe**: "The InferenceService points at the MLflow-registered model's
  artifacts and supports canary rollout through KServe's
  `canaryTrafficPercent` annotation — you patch in a new model version's
  storage path, start it at a small traffic percentage, watch it, then
  ramp up or roll back by changing that one annotation. It's the same
  small-blast-radius rollout pattern as the canary work I've done on AKS,
  just expressed through Knative revisions instead of a service-mesh
  routing rule."
- **Drift exporter / closing the loop**: "This is the self-healing piece —
  Population Stability Index is a standard technique for catching
  distribution shift, and I used real industry threshold values (0.1 /
  0.25), not arbitrary ones. When aggregate drift crosses the alert
  threshold, in production that would fire through Alertmanager into an
  Argo Events sensor that kicks off the Kubeflow pipeline automatically —
  I stubbed that last network call rather than pretend it ran against
  infrastructure I don't have for this interview."
- **RAG auto-RCA**: "Given a new alert, this retrieves the most similar
  past incident and either has Claude synthesize an RCA grounded in that
  postmortem, or falls back to pulling the Root Cause/Fix sections
  directly if no API key is set. Two of the three incident docs are based
  on real production issues I've dealt with — a base image's inherited
  ENTRYPOINT silently swallowing a service's actual binary, and a JDK
  base-image bump breaking Lombok annotation processing — which is why the
  postmortems read as specific and diagnostic rather than generic."
- **Overall framing**: "I don't have production ML engineering experience,
  and I'm not claiming this is production-hardened — what I'm showing is
  that I can read this toolchain's architecture, wire a coherent
  train-serve-monitor-retrain loop across four different tools, and be
  precise about where the real gaps are, which is the same discipline I'd
  bring to operating this stack rather than just building a demo of it."
