
import mocks
import gstreamer.motion_detector as motion_detector
import unittest

class TestMotionDetector(unittest.TestCase):

  def test_get_motion_continuity(self):
    self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

  def test_get_motion_continuity2(self):
    obj = mocks.Object(id=1, score=0.5, bbox=mocks.BBox(xmin=1, ymin=1, xmax=2, ymax=2))

    l = (obj,obj)
    self.assertEqual(motion_detector.get_motion_continuity(obj,l), 5, "Should be 5")

if __name__ == '__main__':
    unittest.main()

