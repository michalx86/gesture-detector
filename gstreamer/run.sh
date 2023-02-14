if false; then
  # Google Coral DevBoard
  VIDEO_DEV="/dev/video1"
  OPTIONS="--edgetpu"
  #OPTIONS="--edgetpu --sync_classification"
  #OPTIONS="--edgetpu --sync_classification --hq_sync_classification"
  #GEST_DET_MODEL="ssd_mobilenet_v2_gestures_edgetpu.tflite"
  GEST_DET_MODEL="efficientdet0-lite-gestures_edgetpu.tflite"
  FACE_CLAS_MODEL="mobilenet_v1_face_classifier_edgetpu.tflite"
else
  # RaspberryPi 4
  export DISPLAY=:0
  VIDEO_DEV="/dev/video0"
  #OPTIONS="--sync_classification"
  OPTIONS="--sync_classification --hq_sync_classification"
  #OPTIONS="--sync_classification --hq_sync_classification --detect_face_only"
  GEST_DET_MODEL="ssd_mobilenet_v2_gestures.tflite"
  GEST_DET_MODEL="efficientdet0-lite-gestures.tflite"
  FACE_CLAS_MODEL="mobilenet_v1_face_classifier.tflite"
fi

MODEL_DIR="../all_models"

GESTURE_DET_LABELS="gesture_det_labels.txt"
FACE_CLAS_LABELS="face_class_labels.txt"

CPE_IP_ARG=""
#CPE_IP_ARG="--cpeip 192.168.0.201"
#CPE_IP_ARG="--cpeip 192.168.0.100"

python3 detect.py \
      --model $MODEL_DIR/$GEST_DET_MODEL   --labels $MODEL_DIR/$GESTURE_DET_LABELS \
      --face_model $MODEL_DIR/$FACE_CLAS_MODEL   --face_labels $MODEL_DIR/$FACE_CLAS_LABELS \
      --videosrc=${VIDEO_DEV} --crop --zoom_factor=1.0 ${OPTIONS} $CPE_IP_ARG
