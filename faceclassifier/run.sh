DATA_DIR="./WebcamFaces"
#DATA_DIR="./FamFaces"

RETRAINED_MODEL_NAME="face_classifier_model"
OUTPUT_DIR="../all_models/"
OUTPUT_MODEL_FILE="${OUTPUT_DIR}${RETRAINED_MODEL_NAME}.tflite"
OUTPUT_LABEL_FILE="${OUTPUT_DIR}${RETRAINED_MODEL_NAME}.txt"
rm ${OUTPUT_MODEL_FILE}  ${OUTPUT_LABEL_FILE}
python3 ./imprinting_learning.py --test_ratio=0.20 --model_path ./mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite --data ${DATA_DIR} --output ${OUTPUT_MODEL_FILE}
