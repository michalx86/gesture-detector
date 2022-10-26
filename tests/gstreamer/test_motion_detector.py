
import tests.gstreamer.mocks as mocks
import gstreamer.motion_detector as motion_detector
import unittest

class TestMotionDetector(unittest.TestCase):

  def test_get_motion_continuity_no_oldobj_no_detections(self):
    self.assertEqual(motion_detector.get_motion_continuity(None,[]), None , "Should be None")

  def test_get_motion_continuity_no_oldobj_1_detection(self):
    obj = mocks.Object(id=1, score=0.5, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))
    exp_obj = mocks.Object(id=1, score=0.0, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))

    new_obj = motion_detector.get_motion_continuity(None,[obj])

    self.assertEqual(new_obj, exp_obj, "Should be equal to: {}".format(exp_obj))
    self.assertEqual(obj.score, 0.5, "Input obj should not be modified.")

  def test_get_motion_continuity_no_oldobj_3_detections(self):
    obj1 = mocks.Object(id=1, score=0.6, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))
    obj2 = mocks.Object(id=2, score=0.8, bbox=mocks.BBox(xmin=3, ymin=3, xmax=4, ymax=4))
    obj3 = mocks.Object(id=3, score=0.7, bbox=mocks.BBox(xmin=5, ymin=5, xmax=6, ymax=6))
    exp_obj = mocks.Object(id=2, score=0.0, bbox=mocks.BBox(xmin=3, ymin=3, xmax=4, ymax=4))

    new_obj = motion_detector.get_motion_continuity(None,[obj1, obj2, obj3])

    self.assertEqual(new_obj, exp_obj, "Should be equal to: {}".format(exp_obj))

  def test_get_motion_continuity_oldobj_no_detection(self):
    obj = mocks.Object(id=1, score=0.5, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))
    exp_obj = mocks.Object(id=1, score=0.5, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))

    new_obj = motion_detector.get_motion_continuity(obj, None)

    self.assertEqual(new_obj, exp_obj, "Should be equal to: {}".format(exp_obj))
    self.assertEqual(obj.score, 0.5, "Input obj should not be modified.")

  def test_get_motion_continuity_oldobj_1_detection_nointersection(self):
    obj1 = mocks.Object(id=1, score=0.6, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))
    obj2 = mocks.Object(id=1, score=0.8, bbox=mocks.BBox(xmin=3, ymin=3, xmax=4, ymax=4))

    new_obj = motion_detector.get_motion_continuity(obj1, [obj2])

    self.assertEqual(new_obj, None, "Should be None")

if __name__ == '__main__':
    unittest.main()

