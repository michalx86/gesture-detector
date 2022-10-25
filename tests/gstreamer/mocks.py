
import collections

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
"""Represents a detected object.

  .. py:attribute:: id

      The object's class id.

  .. py:attribute:: score

      The object's prediction score.

  .. py:attribute:: bbox

      A :obj:`BBox` object defining the object's location.
"""


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
  """The bounding box for a detected object.

  .. py:attribute:: xmin

      X-axis start point

  .. py:attribute:: ymin

      Y-axis start point

  .. py:attribute:: xmax

      X-axis end point

  .. py:attribute:: ymax

      Y-axis end point
  """
  __slots__ = ()

  @property
  def width(self):
    """The bounding box width."""
    return self.xmax - self.xmin

  @property
  def height(self):
    """The bounding box height."""
    return self.ymax - self.ymin

  @property
  def area(self):
    """The bound box area."""
    return self.width * self.height

  @property
  def valid(self):
    """Indicates whether bounding box is valid or not (boolean).

    A valid bounding box has xmin <= xmax and ymin <= ymax (equivalent
    to width >= 0 and height >= 0).
    """
    return self.width >= 0 and self.height >= 0

  def scale(self, sx, sy):
    """Scales the bounding box.

    Args:
      sx (float): Scale factor for the x-axis.
      sy (float): Scale factor for the y-axis.

    Returns:
      A :obj:`BBox` object with the rescaled dimensions.
    """
    return BBox(
        xmin=sx * self.xmin,
        ymin=sy * self.ymin,
        xmax=sx * self.xmax,
        ymax=sy * self.ymax)

  def translate(self, dx, dy):
    """Translates the bounding box position.

    Args:
      dx (int): Number of pixels to move the box on the x-axis.
      dy (int): Number of pixels to move the box on the y-axis.

    Returns:
      A :obj:`BBox` object at the new position.
    """
    return BBox(
        xmin=dx + self.xmin,
        ymin=dy + self.ymin,
        xmax=dx + self.xmax,
        ymax=dy + self.ymax)

  def map(self, f):
    """Maps all box coordinates to a new position using a given function.

    Args:
      f: A function that takes a single coordinate and returns a new one.

    Returns:
      A :obj:`BBox` with the new coordinates.
    """
    return BBox(
        xmin=f(self.xmin),
        ymin=f(self.ymin),
        xmax=f(self.xmax),
        ymax=f(self.ymax))

  @staticmethod
  def intersect(a, b):
    """Gets a box representing the intersection between two boxes.

    Args:
      a: :obj:`BBox` A.
      b: :obj:`BBox` B.

    Returns:
      A :obj:`BBox` representing the area where the two boxes intersect
      (may be an invalid box, check with :func:`valid`).
    """
    return BBox(
        xmin=max(a.xmin, b.xmin),
        ymin=max(a.ymin, b.ymin),
        xmax=min(a.xmax, b.xmax),
        ymax=min(a.ymax, b.ymax))

  @staticmethod
  def union(a, b):
    """Gets a box representing the union of two boxes.

    Args:
      a: :obj:`BBox` A.
      b: :obj:`BBox` B.

    Returns:
      A :obj:`BBox` representing the unified area of the two boxes
      (always a valid box).
    """
    return BBox(
        xmin=min(a.xmin, b.xmin),
        ymin=min(a.ymin, b.ymin),
        xmax=max(a.xmax, b.xmax),
        ymax=max(a.ymax, b.ymax))

  @staticmethod
  def iou(a, b):
    """Gets the intersection-over-union value for two boxes.

    Args:
      a: :obj:`BBox` A.
      b: :obj:`BBox` B.

    Returns:
      The intersection-over-union value: 1.0 meaning the two boxes are
      perfectly aligned, 0 if not overlapping at all (invalid intersection).
    """
    intersection = BBox.intersect(a, b)
    if not intersection.valid:
      return 0.0
    area = intersection.area
    return area / (a.area + b.area - area)

