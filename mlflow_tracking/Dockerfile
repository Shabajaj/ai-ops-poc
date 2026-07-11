# Minimal training image for train_and_register.py — built to run as a
# one-shot Kubernetes Job, not a long-running service.
#
# KNOWN LIMITATION (be upfront about this): this container's default
# tracking URI (sqlite:///mlflow.db) writes to the CONTAINER's own
# filesystem. Since a Job pod's filesystem is thrown away once the pod
# exits, each Job run starts from a blank registry — there's no
# persistence across runs on the cluster the way there is when you run
# this script locally against a file that survives on your own disk.
# A real deployment would either mount a PersistentVolumeClaim at
# /app/mlflow.db's directory, or (better) point MLFLOW_TRACKING_URI at a
# real, separately-deployed MLflow server (see docker-compose.yml for
# what that server looks like) so multiple Job runs share one registry.
# This image supports that via the MLFLOW_TRACKING_URI env var — it's
# just not pointed at one by default, since none is deployed yet.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_and_register.py .

ENTRYPOINT ["python", "train_and_register.py"]
