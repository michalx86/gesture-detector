MODEL_DIR="$HOME/my_models/gestures_edgetpu"

MODEL="efficientdet0-lite-gestures_edgetpu.tflite"
#MODEL="ssd_mobilenet_v2_gestures_edgetpu.tflite"

CPE_IP_ARG=""
#CPE_IP_ARG="--cpeip 192.168.0.201"
#CPE_IP_ARG="--cpeip 192.168.0.100"

python3 detect.py   --model $MODEL_DIR/$MODEL   --labels $MODEL_DIR/labels.txt   --videosrc=/dev/video1 --crop --zoom_factor=1.0 $CPE_IP_ARG
