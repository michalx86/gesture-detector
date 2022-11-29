MODEL_DIR="$HOME/my_models/gestures_edgetpu"

CPE_IP_ARG=""
#CPE_IP_ARG="--cpeip 192.168.0.201"

python3 detect.py   --model $MODEL_DIR/ssd_mobilenet_v2_gestures_edgetpu.tflite   --labels $MODEL_DIR/labels.txt   --videosrc=/dev/video1 --crop --zoom_factor=1.5 $CPE_IP_ARG
