# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo which runs object detection on camera frames using GStreamer.

Run default object detection:
python3 detect.py

Choose different camera and input encoding
python3 detect.py --videosrc /dev/video1 --videofmt jpeg

TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import colorsys
import gstreamer
import os
import requests
import time

from common import avg_fps_counter, SVG
import object_tracker
from key_emitter import KeyEmitter
from key_emitter import KeyCode
from key_emitter import RawCode
from key_emitter import KeyEvent
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from pycoral.adapters import classify
from pycoral.adapters.common import set_input
from PIL import Image

def rgb(color):
    return 'rgb(%s, %s, %s)' % color

def color(i, total):
    return tuple(int(255.0 * c) for c in colorsys.hsv_to_rgb(i / total, 1.0, 1.0))

def make_palette(keys):
    return {key : rgb(color(i, len(keys))) for i, key in enumerate(keys)}

def make_label_colors(labels):
    if labels:
        palette = make_palette(labels.keys())
        return lambda obj_id: palette[obj_id]

    return lambda obj_id: 'white'

OUTLINE_COLOR = 'rgb(255,255,255)'
OUTLINE_WIDTH = 2
LINE_COLOR = 'rgb(200,200,200)'
LINE_WIDTH = 4
FILL_COLOR = 'rgb(255,255,255)'
PROGRESS_BAR_WIDTH_RATIO = 0.1

def calc_coord(bbox, box_x, box_y, scale_x, scale_y):
    # Absolute coordinates, input tensor space.
    x, y = bbox.xmin, bbox.ymin
    w, h = bbox.width, bbox.height
    # Subtract boxing offset.
    x, y = x - box_x, y - box_y
    # Scale to source coordinate space.
    return x * scale_x, y * scale_y, w * scale_x, h * scale_y

def generate_svg(src_size, inference_box, objs, labels, label_colors, text_lines, tracked_obj, face_obj, face_label):

    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        bbox = obj.bbox
        if not bbox.valid:
            continue
        x, y, w, h = calc_coord(bbox, box_x, box_y, scale_x, scale_y)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        svg.add_text(x, y - 5, label, 20)
        if obj == face_obj and face_label is not None:
            svg.add_text(x,y - 25, "Hi "+ face_label + "!", 20)
        svg.add_rect(x, y, w, h, label_colors(obj.id), LINE_WIDTH, obj.score)

    if tracked_obj is not None:
        bbox = tracked_obj.bbox
        if bbox.valid:
            x, y, w, h = calc_coord(bbox, box_x, box_y, scale_x, scale_y)
            progressbar_width = w * tracked_obj.score - 2 * LINE_WIDTH
            progressbar_height = h * PROGRESS_BAR_WIDTH_RATIO
            svg.add_rect(x + LINE_WIDTH, y + LINE_WIDTH,
                         progressbar_width, progressbar_height, 
                         LINE_COLOR, LINE_WIDTH, 1.0, FILL_COLOR)
            svg.add_rect(x + OUTLINE_WIDTH+1, y + OUTLINE_WIDTH+1,
                         w - 2 * (OUTLINE_WIDTH + 1),  progressbar_height + OUTLINE_WIDTH, 
                         OUTLINE_COLOR, OUTLINE_WIDTH, 1.0)

    return svg.finish()

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--crop', help='Input video should be cropped to aspect ratio 1:1',
                        action='store_true')
    parser.add_argument('--zoom_factor', type=float, default=1.0,
                        help='Additional zoom when crop is used')
    parser.add_argument('--cpeip', help='IP address of CPE to send keycodes to')
    parser.set_defaults(crop=False)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    label_colors = make_label_colors(labels)

    interpreter_classifier = make_interpreter('/home/mendel/edgetpu/retrain-imprinting/retrained_imprinting_model.tflite')
    face_labels = read_label_file('/home/mendel/edgetpu/retrain-imprinting/retrained_imprinting_model.txt')
    interpreter_classifier.allocate_tensors()
    classifier_size = input_size(interpreter_classifier)
    face_label = None

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    tracked_obj = None
    key_emtr = KeyEmitter()

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal fps_counter
      nonlocal tracked_obj
      nonlocal key_emtr
      nonlocal face_label
      start_time = time.monotonic()

      run_inference(interpreter, input_tensor)
      # For larger input image sizes, use the edgetpu.classification.engine for better performance
      objs = get_objects(interpreter, args.threshold)[:args.top_k]

      end_time = time.monotonic()

      face_obj = None
      face_objs = list(filter( lambda obj: obj.score > 0.3 and obj.id == 10, objs))
      face_coords = 0.0, 0.0, 0.01
      if face_objs:
          face_obj = face_objs[0]
          #print('Face: {}'.format(face_objs[0]))
          x1, y1, x2, y2 = face_objs[0].bbox
          w = x2 - x1
          h = y2 - y1
          if w > h:
              l = w
              y1 = y1 - (l - h) // 2
          else:
              l = h
              x1 = x1 - (l - w) // 2

          inference_box_size = inference_box[2]
          x = x1 / inference_box_size
          y = y1 / inference_box_size
          l = l / inference_box_size
          face_coords = x,y,l


      filtered_objs = list(filter( lambda obj: obj.score > 0.5 and obj.id <= RawCode.GEST_PAUSE.value, objs))
      tracked_obj = object_tracker.track(tracked_obj, filtered_objs)

      tracked_obj_id    = tracked_obj.id    if tracked_obj is not None else -1
  
      key,event,fill = key_emtr.push_input(tracked_obj_id, end_time)
      if tracked_obj is not None:
          tracked_obj = tracked_obj._replace(score = fill)

      if key != KeyCode.NO_KEY and event != KeyEvent.RELEASE:
          print("Key: {}, Event {}".format(key, event))
          if args.cpeip is not None:
              cpe_command = 'http://{}:10014/keyinjector/emulateuserevent/{}/{}'.format(args.cpeip, hex(key.value), KeyEvent.PRESS_RELEASE.value) 
              # alternatively we could use event.value, but it is not intuitive with sign language
              print(cpe_command)
              response = requests.get(cpe_command)
              print(response.text)

      text_lines = [
          'Inference: {:_>3.0f} ms'.format((end_time - start_time) * 1000),
          'FPS: {} fps'.format(round(next(fps_counter))),
          #'ID: {}'.format(tracked_obj_id if tracked_obj_id != -1 else '--'),
          #'Score: {}'.format(tracked_obj_score),
      ]
      #print(' '.join(text_lines))

      return generate_svg(src_size, inference_box, objs, labels, label_colors, text_lines, tracked_obj, face_obj, face_label), face_coords

    def user_classifier_callback(input_tensor):
      nonlocal face_label
      #set_input(interpreter_classifier, classifier_img)
      #interpreter_classifier.invoke()
      run_inference(interpreter_classifier, input_tensor)
      candidates = classify.get_classes(interpreter_classifier, 5, score_threshold=0.95)
      if candidates:
          face_label = face_labels[0]
      for candidate in candidates:
        #print("Candidate {}".format(candidate))
        print("Candidate Name {}".format(face_labels[candidate.id]))
      return

    result = gstreamer.run_pipeline(user_callback,
                                    user_classifier_callback,
                                    src_size=(640, 480), # (640,480) or (800, 600) or (1280,720)
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt,
                                    crop=args.crop,
                                    zoom_factor=args.zoom_factor)

if __name__ == '__main__':
    main()
