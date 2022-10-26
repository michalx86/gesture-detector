

#def by_score(obj):
#    return obj.score

def get_motion_continuity(old_obj, objs):
  #objs.sort(key=by_score)

  new_obj = old_obj

  if objs:
    if new_obj is None:
      new_obj = objs[0]
      for obj in objs:
        if new_obj.score < obj.score:
          new_obj = obj
#    else:
#      fitness = 0;
#      for obj in objs:
#        iou(new_obj, obj)

  if new_obj != old_obj:
    new_obj = new_obj._replace(score = 0)

  return new_obj
