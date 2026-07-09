# Incident 001: gomarketdata service — CrashLoopBackOff after image rebuild

**Severity:** Sev2
**Service:** gomarketdata
**Status:** Resolved

## Summary
After a routine base-image rebuild, the `gomarketdata` deployment entered
CrashLoopBackOff on all replicas within minutes of rollout. No error output
appeared in application logs — containers simply started and exited almost
immediately with exit code 0, giving no indication of an application-level
failure.

## Symptoms
- All pods for the deployment cycling through CrashLoopBackOff
- `kubectl logs` showed no stack trace, no panic, no application log lines at
  all — as if the app never actually started
- Exit code 0 on every restart (not a crash in the traditional sense — the
  process was exiting cleanly, just immediately)
- Previous image tag, redeployed manually, worked fine — pointed at the new
  image build, not at cluster/config drift

## Root Cause
The team had recently switched to a new internal base image maintained by a
platform team. That base image defined its own Docker `ENTRYPOINT` (a generic
init/wrapper script intended for a different class of services). Because the
`gomarketdata` Dockerfile only set a `CMD`, not its own `ENTRYPOINT`, Docker's
layering rules meant the inherited `ENTRYPOINT` from the base image silently
took over — the wrapper script ran, did nothing relevant for this service,
and exited 0. The application binary was never actually invoked.

This is a classic Docker footgun: `ENTRYPOINT` and `CMD` interact such that a
child image's `CMD` is passed as *arguments to the parent's ENTRYPOINT*
unless the child image explicitly overrides `ENTRYPOINT` too. Nothing about
this shows up as a build error or a Kubernetes-level error — the container
starts, runs whatever ENTRYPOINT resolves to, and exits. From Kubernetes'
point of view this looks identical to a normal (if fast) process exit,
which is why it manifests purely as CrashLoopBackOff with no logs.

## Fix
Added an explicit `ENTRYPOINT ["/app/gomarketdata"]` to the service's own
Dockerfile, so it no longer inherits the base image's ENTRYPOINT. Verified
by inspecting the built image with `docker inspect --format='{{.Config.Entrypoint}} {{.Config.Cmd}}'`
before and after, and by running the image locally with `docker run` to
confirm the actual binary executed and produced expected startup logs.

## Detection Gap
Nothing paged on "zero log lines emitted" — the alert that fired was the
generic CrashLoopBackOff/restart-count alert, which gave no hint that the
root cause was in image layering rather than application code. Added a
follow-up check (not fully implemented) to flag deployments where restart
count is high but log volume in the same window is near zero, since that
combination is a strong signal for "container never really started" rather
than "application crashed."

## Prevention
- New internal base images now require a documented `ENTRYPOINT`/`CMD`
  contract in their README, and downstream Dockerfiles are expected to
  either explicitly override `ENTRYPOINT` or explicitly rely on the
  documented one — no more silent inheritance
- Added a CI smoke-test step that runs the built image and asserts at least
  one expected startup log line appears within N seconds, catching this
  class of bug before it reaches a cluster
