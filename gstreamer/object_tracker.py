from pycoral.adapters.detect import BBox

def track(old_tracked_obj, objs):
    new_tracked_obj = None
    detection_is_same = False

    if objs:
        if old_tracked_obj is not None:
            best_fitness = 0.0;
            for obj in objs:
                fitness = BBox.iou(old_tracked_obj.bbox, obj.bbox)
                if fitness > 0.0:
                    if old_tracked_obj.id != obj.id:
                        new_tracked_obj = None
                        break
                    elif best_fitness < fitness:
                        new_tracked_obj = obj
                        best_fitness = fitness
                        detection_is_same = True

        if new_tracked_obj is None:
            new_tracked_obj = objs[0]
            for obj in objs:
                if new_tracked_obj.score < obj.score:
                    new_tracked_obj = obj

    if new_tracked_obj is not None:
        if detection_is_same:
            new_score = old_tracked_obj.score + 1
        else:
            new_score = 0
        new_tracked_obj = new_tracked_obj._replace(score = new_score)

    return new_tracked_obj
