#from pycoral.adapters.detect import BBox
from tests.gstreamer.mocks import BBox 


def get_motion_continuity(old_obj, objs):
  new_obj = old_obj

  if objs:
    if new_obj is None:
      new_obj = objs[0]
      for obj in objs:
        if new_obj.score < obj.score:
          new_obj = obj
    else:
      fitness = 0;
      for obj in objs:
          BBox.iou(new_obj.bbox, obj.bbox)

  if new_obj != old_obj:
    new_obj = new_obj._replace(score = 0)

  return new_obj
