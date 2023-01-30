MODEL_DIR="../all_models"

#GEST_DET_MODEL="ssd_mobilenet_v2_gestures_edgetpu.tflite"
GEST_DET_MODEL="efficientdet0-lite-gestures_edgetpu.tflite"
GESTURE_DET_LABELS="gesture_det_labels.txt"

FACE_CLAS_MODEL="face_classifier_model.tflite"
FACE_CLAS_LABELS="face_classifier_model.txt"

CPE_IP_ARG=""
#CPE_IP_ARG="--cpeip 192.168.0.201"
#CPE_IP_ARG="--cpeip 192.168.0.100"

python3 detect.py \
      --model $MODEL_DIR/$GEST_DET_MODEL   --labels $MODEL_DIR/$GESTURE_DET_LABELS \
      --face_model $MODEL_DIR/$FACE_CLAS_MODEL   --face_labels $MODEL_DIR/$FACE_CLAS_LABELS \
      --videosrc=/dev/video1 --crop --zoom_factor=1.0 $CPE_IP_ARG
