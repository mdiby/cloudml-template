#!/bin/bash

REGION=europe-west1
BUCKET=ksalama-gcs-cloudml

MODEL_NAME="synth_regressor"
MODEL_VERSION="v1"

MODEL_BINARIES=$(gsutil ls gs://${BUCKET}/cloudml-template/examples/synthetic-regression/trained_models/${MODEL_NAME}/export/Servo | tail -1)

gsutil ls ${MODEL_BINARIES}

# delete model version
gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}

# delete model
gcloud ml-engine models delete ${MODEL_NAME}

# deploy model to GCP
gcloud ml-engine models create ${MODEL_NAME} --regions ${REGION}

# deploy model version
gcloud ml-engine versions create ${MODEL_VERSION} --model=${MODEL_NAME} --origin ${MODEL_BINARIES} --runtime-version=1.2

# invoke deployed model to make prediction given new data instances
gcloud ml-engine predict --model=${MODEL_NAME} --version=${MODEL_VERSION} --json-instances=data/new-data.json