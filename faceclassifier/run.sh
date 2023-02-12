DATA_DIR="./WebcamFaces"
#DATA_DIR="./FamFaces"

EDGE_TPU=false
if ${EDGE_TPU}; then
    SRC_MODEL_NAME="mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"
    RETRAINED_MODEL_NAME="mobilenet_v1_face_classifier_edgetpu"
    OPTIONS="--edgetpu"
else
    SRC_MODEL_NAME="mobilenet_v1_l2norm_quant.tflite"
    RETRAINED_MODEL_NAME="mobilenet_v1_face_classifier"
    OPTIONS=""
fi
echo "Source Model Name: "${SRC_MODEL_NAME}
echo "Retrained Model Name: "${RETRAINED_MODEL_NAME}

RETRAINED_MODEL_LABELS="face_class_labels"
OUTPUT_DIR="../all_models/"

OUTPUT_MODEL_FILE="${OUTPUT_DIR}${RETRAINED_MODEL_NAME}.tflite"
TMP_LABEL_FILE="${OUTPUT_DIR}${RETRAINED_MODEL_NAME}.txt"
OUTPUT_LABEL_FILE="${OUTPUT_DIR}${RETRAINED_MODEL_LABELS}.txt"

rm ${OUTPUT_MODEL_FILE}  ${OUTPUT_LABEL_FILE}
python3 ./imprinting_learning.py --test_ratio=0.20 --model_path ${SRC_MODEL_NAME} --data ${DATA_DIR} --output ${OUTPUT_MODEL_FILE} ${OPTIONS}
mv ${TMP_LABEL_FILE} ${OUTPUT_LABEL_FILE}
