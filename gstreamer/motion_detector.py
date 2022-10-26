#from pycoral.adapters.detect import BBox
from tests.gstreamer.mocks import BBox 


def get_motion_continuity(old_detection_obj, objs):
    new_detection_obj = None
    detection_is_same = False

    if objs:
        if old_detection_obj is not None:
            best_fitness = 0.0;
            for obj in objs:
                fitness = BBox.iou(old_detection_obj.bbox, obj.bbox)
                if fitness > 0.0:
                    if old_detection_obj.id != obj.id:
                        new_detection_obj = None
                        break
                    elif best_fitness < fitness:
                        new_detection_obj = obj
                        best_fitness = fitness
                        detection_is_same = True

        if new_detection_obj is None:
            new_detection_obj = objs[0]
            for obj in objs:
                if new_detection_obj.score < obj.score:
                    new_detection_obj = obj

    if new_detection_obj is not None:
        if detection_is_same:
            new_score = old_detection_obj.score + 1
        else:
            new_score = 0
        new_detection_obj = new_detection_obj._replace(score = new_score)

    return new_detection_obj
