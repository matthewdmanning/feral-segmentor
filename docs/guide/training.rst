Training
========

Local training
--------------

The canonical training entrypoint is ``feral_vision.training.trainer``, wired
via Hydra:

.. code-block:: bash

   uv run python -m feral_vision.training.trainer

This builds the model (:mod:`feral_vision.models.register_model`), optimizer,
scheduler, and loss function from ``conf/train/`` (see :doc:`../api/training`),
then runs :meth:`~feral_vision.training.trainer.Trainer.fit`. Metrics are
logged to MLflow when a run is active. MLflow uploads only the selected best
model artifact; intermediate checkpoints remain local and are not retained in
the artifact store.

GCP training
------------

Requires ``GCP_PROJECT``, ``GCS_BUCKET``, and the Cloud Run MLflow endpoint:

.. code-block:: bash

   GCP_PROJECT=my-proj GCS_BUCKET=my-bucket \
   MLFLOW_TRACKING_URI=https://mlflow-abc-uc.a.run.app \
   bash scripts/gcp_train.sh

Data pipeline
-------------

DVC owns data preparation only (fetch, preprocess, augment) — not training or
evaluation:

.. code-block:: bash

   dvc repro
