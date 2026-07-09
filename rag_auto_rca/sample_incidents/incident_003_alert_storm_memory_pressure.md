# Incident 003: Multi-service alert storm from node-level memory pressure

**Severity:** Sev1
**Service:** multiple (cascading)
**Status:** Resolved
**Note:** This incident is synthetic (constructed for this POC), unlike
incidents 001 and 002 which are based on real past incidents. It's included
to give the RAG retrieval step a "noisy, cascading, multi-service" incident
shape to match against, distinct from the other two single-service cases.

## Summary
A slow memory leak in a sidecar container (log-shipping agent) on a shared
node gradually consumed available node memory over several hours. Once node
memory pressure crossed the eviction threshold, the kubelet began evicting
pods across multiple unrelated services co-located on that node, producing
a burst of simultaneous alerts (pod evicted, pod restart, latency spikes,
error rate spikes) across services that had no functional relationship to
each other — making the incident initially look like several unrelated
outages instead of one shared-node root cause.

## Symptoms
- Alerts fired for 5+ unrelated services within a ~10 minute window
- Each affected service's own dashboards showed restarts/evictions but
  nothing wrong in that service's own code path, config, or recent deploys
- Node-level memory usage graph showed a slow, steady climb over ~6 hours
  before crossing the eviction threshold — visible in hindsight, not
  something any single service's on-call was watching
- `kubectl describe node` on the affected node showed `MemoryPressure=True`
  and multiple `Evicted` pods with reason `The node was low on resource: memory`

## Root Cause
A log-shipping sidecar injected via a shared PodTemplate/admission webhook
had a memory leak triggered by a specific log-line pattern that appeared
more frequently under a recent traffic mix change. Because the sidecar was
injected identically across many services on the same node pool, the leak
accumulated per-pod but the *symptom* (node memory pressure, then eviction)
appeared at the node level, cutting across service boundaries. On-call
engineers for each individually affected service had no visibility into the
sidecar or the node-level memory trend — each only saw their own pod get
evicted, which looks identical to a resource-limit misconfiguration or a
demand spike in that service alone.

## Fix
- Immediate: cordoned the affected node and rescheduled pods onto healthy
  nodes to stop further evictions
- Root cause: patched the log-shipping sidecar to fix the leak (a buffer
  that wasn't being flushed/released for a specific log format), and added
  a memory limit + restart policy on the sidecar itself so a future leak in
  it fails contained instead of pressuring the whole node

## Detection Gap
Alerting was entirely per-service (restart count, latency, error rate) —
there was no alert on the actual leading indicator, node-level memory trend
over hours, until it had already crossed into eviction territory and become
a multi-service incident. Individual service on-call had no reason to
suspect a shared node/sidecar cause from their own dashboards alone.

## Prevention
- Added a node-level memory trend alert (rate of increase over a rolling
  window, not just an absolute threshold) so slow leaks are visible before
  they cause evictions
- Added resource limits to the shared sidecar so a leak in it can no longer
  consume unbounded node memory
- This incident is the motivating example for `monitoring/drift_exporter.py`
  in this POC: the same "watch a rolling trend against a baseline and flag
  before it becomes a hard failure" pattern applies to model prediction
  drift, not just node memory
