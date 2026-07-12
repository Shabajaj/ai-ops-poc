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
- **`kserve/inferenceservice.yaml`** — deployed live on the same EKS
  cluster with a real KServe (v0.19.0) + Knative Serving + Istio install
  (Serverless mode, needed for real weighted canary traffic — RawDeployment
  mode doesn't support it). Verified: the predictor pod running `2/2`,
  a real V2-protocol inference request returning correct predictions for
  both a high-risk input (`[5 restarts, 87% mem, 2 pull failures, 1.5h
  since config change]` → `1`) and a clearly low-risk one (`[0, 15%, 0,
  72h]` → `0`), and a real canary rollout — two revisions serving traffic
  simultaneously at a genuine 90/10 split, then shifted live to 50/50 by
  patching one field. Two real bugs were hit and fixed along the way
  (not just designed around): a training/serving library version-skew
  breaking model loading, and an outdated understanding of *how* KServe's
  canary mechanism actually works in this version — both documented
  in-line in `inferenceservice.yaml` with what the fix was and how it was
  confirmed.
- **`kubeflow/`** — `training_pipeline.py` still compiles cleanly to
  `training_pipeline.yaml` (KFP's own IR format), but that file needs the
  full Kubeflow Pipelines backend (API server + MySQL + MinIO, plus an
  EBS CSI driver this cluster doesn't have) to actually run — a bigger
  lift than this timeline could absorb after Items 1 and 3's infra
  debugging. Instead, `training_pipeline_argo_workflow.yaml` is a
  hand-written native Argo Workflow mirroring the same DAG shape, run
  live on a real Argo Workflows install on the same cluster: all 4 steps
  completed, `evaluate-model` independently scored F1=0.5954, and the
  conditional `register-and-promote` step correctly triggered and
  promoted a version. `argo-executor-rbac.yaml` documents two real RBAC
  gaps in Argo's base install found by submitting and watching it fail.
- Everything else in this repo is still at the status described inline in
  each component's own comments (some fully tested locally, some
  designed-but-not-deployed) — not yet re-verified against a live cluster.

## Infrastructure notes learned from deploying Items 1, 3, and 4 live

Both of these came from actually running things on a real cluster, not
from reading docs beforehand — worth knowing if this comes up in an
interview as "what surprised you":

- **AWS account Free Tier restriction**: this AWS account rejects
  non-Free-Tier-eligible instance types outright (`t3.medium` → hard
  `InvalidParameterCombination` error). Free-Tier-eligible options were
  checked directly via `aws ec2 describe-instance-types --filters
  Name=free-tier-eligible,Values=true` rather than guessed;
  `m7i-flex.large` (same 2 vCPUs as `t3.small`, 4x the memory) was the
  best fit found.
- **EKS's default node security group doesn't allow arbitrary
  control-plane-to-pod traffic.** The `terraform-aws-modules/eks` module
  opens a curated list of common webhook ports (443, 4443, 6443, 8443,
  9443, 10250, 10251) by default — enough for cert-manager (which
  deliberately defaults to port 10250 for this exact reason) but not for
  Istio's istiod webhook (port 15017). Without an explicit security group
  rule for it, admission webhook calls silently time out — diagnosed via
  istiod's own "dummy invalid config not rejected" self-check retry loop,
  not an obvious error message.
- **PodDisruptionBudgets can deadlock a node rolling-update.** Single-
  replica deployments (istiod, Knative's activator/webhook) shipped with
  PDBs requiring `minAvailable` that their own single replica can never
  satisfy during eviction — blocking a node group's rolling update
  indefinitely until the PDBs were removed.
- **`disk_size` on the EKS module is silently ignored** once a custom
  launch template is in play (needed here for node taints/labels) — the
  module's own docs say so, but it's easy to set it, see `terraform plan`
  show no changes, and assume it "just doesn't apply" rather than that
  the setting is a no-op. `block_device_mappings` is the form that
  actually takes effect.
- **`kubectl apply` has a hidden 256KB ceiling** via the
  `last-applied-configuration` annotation it writes — Argo Workflows'
  CRDs (large embedded OpenAPI schemas) exceed it outright. Fixed with
  `kubectl apply --server-side`, which doesn't store that annotation.
- **A tool's base install manifest isn't always self-sufficient RBAC.**
  Argo Workflows' `install.yaml` grants its controller everything it
  needs, but not what individual workflow step pods need to report their
  own results back (`workflowtaskresults` create/patch) or to offload
  large parameters to a ConfigMap — both had to be granted separately,
  found by submitting a real workflow and reading the exact permission
  error twice, not by anticipating it from the docs.
