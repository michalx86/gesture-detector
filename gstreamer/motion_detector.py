#from pycoral.adapters.detect import BBox
from tests.gstreamer.mocks import BBox 


def get_motion_continuity(old_detection_obj, objs):
  new_detection_obj = None

  if objs:
    if old_detection_obj is None:
      new_detection_obj = objs[0]
      for obj in objs:
        if new_detection_obj.score < obj.score:
          new_detection_obj = obj
    else:
      best_fitness = 0;
      for obj in objs:
        fitness = BBox.iou(old_detection_obj.bbox, obj.bbox)
        if best_fitness < fitness:
          new_detection_object = obj
          best_fitness = fitness

  if new_detection_obj is not None and new_detection_obj != old_detection_obj:
    new_detection_obj = new_detection_obj._replace(score = 0)

  return new_detection_obj
