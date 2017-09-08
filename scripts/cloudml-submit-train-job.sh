#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION=europe-west1
TIER=BASIC # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET=ksalama-gcs-cloudml

MODEL_NAME="synth_regressor"

PACKAGE_PATH=trainer
TRAIN_FILES=gs://${BUCKET}/data/synthetic-regression/train-data.csv
VALID_FILES=gs://${BUCKET}/data/synthetic-regression/valid-data.csv
MODEL_DIR=gs://${BUCKET}/trained-models/${MODEL_NAME}

#remove model directory, if you don't want to resume training, or if you have changed the model structure
gsutil -m rm -r ${MODEL_DIR}

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}]
#JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE}

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --runtime-version=1.2 \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        #--config=hyperparams.yaml \
        -- \
        --train-files=${TRAIN_FILES} \
        --num-epochs=10 \
        --train-batch-size=500 \
        --eval-files=${VALID_FILES} \
        --eval-batch-size=500 \
        --learning-rate=0.001 \
        --hidden-units="128,40,10" \
        --layer-sizes-scale-factor=0.5 \
        --num-layers=3 \
        --job-dir=${MODEL_DIR}




