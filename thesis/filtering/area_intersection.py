import polars as pl

from numba import guvectorize, float64, bool


@guvectorize([(float64, float64, float64[:], float64[:], float64[:], float64[:], bool[:])],
             "(),(),(n),(n),(n),(n)->()")
def coords_intersect_polygon(x, y, p1, p2, p3, p4, res) -> bool:
    """
    Checks whether the point given by x,y is inside (or in the edges) of the shape denoted by the
    given points.
    The points must be given in counter-clockwise order.
    """
    # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
    d1 = (p2[0] - p1[0]) * (y - p1[1]) - (x - p1[0]) * (p2[1] - p1[1])
    d2 = (p3[0] - p2[0]) * (y - p2[1]) - (x - p2[0]) * (p3[1] - p2[1])
    d3 = (p4[0] - p3[0]) * (y - p3[1]) - (x - p3[0]) * (p4[1] - p3[1])
    d4 = (p1[0] - p4[0]) * (y - p4[1]) - (x - p4[0]) * (p1[1] - p4[1])
    inside = d1 >= 0 and d2 >= 0 and d3 >= 0 and d4 >= 0
    res[0] = inside


def points_are_counterclockwise(points: list[tuple[float, float]]) -> bool:
    # https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    # Get the index of the point with minimum y and maximum x
    min_point_index = 0
    for i, point in enumerate(points):
        # If y is smaller than the current min
        if point[1] < points[min_point_index][1]:
            min_point_index = i
        # If y is equal to current min
        elif point[1] == points[min_point_index][1]:
            # If x is larger than current min point x
            if point[0] > points[min_point_index][0]:
                min_point_index = i
    a = points[min_point_index-1]
    b = points[min_point_index]
    c = points[(min_point_index+1) % len(points)]
    det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    if det > 0:
        return True
    else:
        return False
