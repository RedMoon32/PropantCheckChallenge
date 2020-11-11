import numpy as np
import cv2
from preprocess import cut_image

AVG_H = 1381
AVG_W = 2272

def get_mask(source:np.array) -> np.array:
    """
    Convert to gray, apply median blur and binary threshold
    Args:
        source: np.array
    Returns:
        tresholded: np.array
    """
    img = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    blured = cv2.medianBlur(img, 5)
    tresholded = np.invert(cv2.threshold(
                           blured, 127, 255, cv2.THRESH_BINARY)[1])
    return tresholded

def count_circles(source:np.array, avg_circle_radius:float) -> int:
    """
    Counts circles by dividing number of black pixels to average circle area.
    Args:
        source: np.array
        avg_circle_radius: float
    Returns:
        circle_count: int
    """
    tresholded = get_mask(source)
    approx_area = np.pi * (avg_circle_radius ** 2)
    circle_count = round(np.sum(tresholded / 255) / approx_area)
    return circle_count


def draw_hough(draw_source:np.array) -> (float, np.array):
    """
    Looks for circles in image, draws them and returns average radius
    and list of centers and radiuses of each circle
    Args:
        draw_source: np.array
    Returns:
        avg_r: float - mean radius of found circles
        circles: list
    """
    tresholded = get_mask(draw_source)

    circles = cv2.HoughCircles(
        tresholded,
        cv2.HOUGH_GRADIENT,
        1,
        30,
        param1=50,
        param2=10,
        minRadius=1,
        maxRadius=17,
    )
    circles = np.uint16(np.around(circles))
    avg_r = np.array(circles[0][:,2]).mean()

    return avg_r, circles[0]


def get_granule_count(processed_img:np.array) -> (int, np.array):
    """
    Returns approximate count of circles in image,
    and distribution of circles' radiuses
    """
    proc = cut_image(
        processed_img,
        30
    )
    
    resized = cv2.resize(proc, (AVG_W, AVG_H))
    avg_r, circles = draw_hough(resized, ret_circles=True)
    found = count_circles(proc, avg_r)
    radiuses = np.array([circle[2] for circle in circles])

    distros = np.array(
        [np.where(radiuses == i)[0].shape[0] for i in range(1, 30)])

    return found, distros / np.sum(distros)
