from typing import Sequence, Optional, Iterable
from matplotlib import colormaps

import numpy as np

from thesis.model.enums import Location


def get_sqlen(location: Location) -> int:
    """
    Length of a 10-metre square in pixels
    """
    match location:
        case Location.RIDDARHUSKAJEN:
            return 477
        case Location.RIDDARHOLMSBRON_N:
            return 942
        case Location.RIDDARHOLMSBRON_S:
            return 939
        case _:
            raise ValueError("Invalid location")


def get_zero_point(location: Location) -> tuple[int, int]:
    """
    Zero point coordinates
    """
    match location:
        case Location.RIDDARHUSKAJEN:
            return 599, 506
        case Location.RIDDARHOLMSBRON_N:
            return 1156, 1008
        case Location.RIDDARHOLMSBRON_S:
            return 497, 1349
        case _:
            raise ValueError("Invalid location")


def get_image_location(location: Location) -> str:
    from thesis.files import BASE_PATH
    base = BASE_PATH.joinpath("data/docs/")
    match location:
        case Location.RIDDARHUSKAJEN:
            return base.joinpath("1_Riddarhuskajen_coordinate_system.png")
        case Location.RIDDARHOLMSBRON_N:
            return base.joinpath("2_RiddarholmsbronN_coordinate_system.png")
        case Location.RIDDARHOLMSBRON_S:
            return base.joinpath("3_RiddarholmsbronS_coordinate_system.png")
        case _:
            raise ValueError("Invalid location")


# The below code is especially riddled with spaghetti calculations

# The transform_xx functions transform image coordinates to x,y coordinates.
# Since the coordinates were initially extracted with wolfram, the (0,0) position is on the bottom left of the image.

# The location_to_image_xx functions print xy coordinates into the image.

def transform_riddarhuskajen(img_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # Length in pixels of an area of size 10 in the region's coordinate system
    square_length = 477
    zero_point = (600, 505)

    transformed_points = []
    for point in img_coords:
        # 10/square_length because square length pixels are 10 units in the coordinate system
        transformed_x = (point[0] - zero_point[0]) * 10 / square_length
        transformed_y = (point[1] - zero_point[1]) * 10 / square_length
        transformed_points.append((transformed_x, transformed_y))
    return transformed_points


def transform_riddarholmsbron_n(img_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    square_length = 941
    zero_point = (1155, 1007)

    transformed_points = []
    for point in img_coords:
        # 10/square_length because square length pixels are 10 units in the coordinate system
        transformed_x = (point[0] - zero_point[0]) * 10 / square_length
        transformed_y = (point[1] - zero_point[1]) * 10 / square_length
        transformed_points.append((transformed_x, transformed_y))
    return transformed_points


def transform_riddarholmsbron_s(img_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    square_length = 939
    zero_point = (498, 1350)

    transformed_points = []
    for point in img_coords:
        # 10/square_length because square length pixels are 10 units in the coordinate system
        transformed_x = (point[0] - zero_point[0]) * 10 / square_length
        transformed_y = (point[1] - zero_point[1]) * 10 / square_length
        transformed_points.append((transformed_x, transformed_y))
    return transformed_points


# In the functions below the zero point is defined relative to the top-right of the image

def _xy_to_image_coords(x: np.ndarray[np.float64], y: np.ndarray[np.float64],
                        zero_x: float, zero_y: float, sqlen: int) -> tuple[
    np.ndarray[np.float64], np.ndarray[np.float64]]:
    new_x = sqlen / 10 * x + zero_x
    new_y = sqlen / 10 * y + zero_y
    return new_x, new_y


def _points_to_image(x: np.ndarray[np.float64], y: np.ndarray[np.float64],
                     w: Sequence[float],
                     zero_point: tuple[int, int], sqlen: int,
                     image_location: str) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_location) as img:
        transformed_x, transformed_y = _xy_to_image_coords(x, y, zero_point[0], zero_point[1], sqlen)
        draw = ImageDraw.Draw(img)
        for x, y, weight in zip(transformed_x, transformed_y, w):
            draw.circle((x, y), fill="red", radius=weight)
        img.show()


def _lines_to_image(lines: Iterable[tuple[np.ndarray[np.float64], np.ndarray[np.float64]]],
                    zero_point: tuple[int, int], sqlen: int,
                    image_location: str) -> None:
    """
    
    :param lines: Collection of pairs of lists of x and y coordinates
    :param zero_point: 
    :param sqlen: 
    :param image_location: 
    :return: 
    """
    from PIL import Image, ImageDraw

    colour = ["red", "green", "blue"]

    with Image.open(image_location) as img:
        for colour, (x, y) in zip(colour, lines):
            transformed_x, transformed_y = _xy_to_image_coords(x, y, zero_point[0], zero_point[1], sqlen)
            draw = ImageDraw.Draw(img)
            draw.line(list(zip(transformed_x, transformed_y)), fill=colour, width=3)
        img.show()


def _polygons_to_image(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                       w: Sequence[float],
                       zero_point: tuple[int, int], sqlen: int,
                       image_location: str) -> None:
    from PIL import Image, ImageDraw

    w = np.array(w)
    w /= np.max(w)

    cmap = colormaps["Greens"]
    colours = (cmap(w) * 255).astype(np.uint8)

    with Image.open(image_location) as img:
        draw = ImageDraw.Draw(img, "RGBA")
        for i, (poly_x, poly_y) in enumerate(zip(x, y)):
            this_fill = tuple(colours[i])
            transformed_x, transformed_y = _xy_to_image_coords(np.array(poly_x), np.array(poly_y),
                                                               zero_point[0], zero_point[1], sqlen)
            draw.polygon(list(zip(transformed_x, transformed_y)), width=1, fill=this_fill)
        img.show()


def polygons_to_image(location: Location, x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                      w: Optional[Sequence[float]] = None) -> None:
    """
    
    :param location: 
    :param x: List of lists containing the x points for each polygon
    :param y: List of lists containing the y points for each polygon
    :return: 
    """
    if w is None:
        w = [1] * len(x)
    square_length = get_sqlen(location)
    zero_point = get_zero_point(location)
    image_location = get_image_location(location)
    _polygons_to_image(x, y, w, zero_point, square_length, image_location)


def points_to_image(location: Location, x: Sequence[float], y: Sequence[float],
                    w: Optional[Sequence[float]] = None) -> None:
    """
    
    :param location: 
    :param x: X coordinates for each point 
    :param y: Y coordinates for each point 
    :return: 
    """
    if w is None:
        w = [1] * len(x)
    square_length = get_sqlen(location)
    zero_point = get_zero_point(location)
    image_location = get_image_location(location)
    _points_to_image(np.array(x), np.array(y), w, zero_point, square_length, image_location)


def lines_to_image(location: Location, lines: Iterable[tuple[Sequence[float], Sequence[float]]]) -> None:
    """

    :param location: 
    :param lines: Collection of pairs of x and y coordinates
    :return: 
    """
    square_length = get_sqlen(location)
    zero_point = get_zero_point(location)
    image_location = get_image_location(location)
    lines_np = [(np.array(x), np.array(y)) for x, y in lines]
    _lines_to_image(lines_np, zero_point, square_length, image_location)
