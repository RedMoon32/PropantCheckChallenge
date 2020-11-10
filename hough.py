import numpy as np
import cv2
from preprocess import cut_image


def count_circles(source, avg_circle_radius):
    """
    Counts circles by dividing number of black pixels to average circle area.
    Args:
        source: np.array
        avg_circle_radius: float
    Returns:
        circle_count: int
    """
    img = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    blured = cv2.medianBlur(img, 5)
    tresholded = np.invert(cv2.threshold(
                           blured, 127, 255, cv2.THRESH_BINARY)[1])
    approx_area = np.pi * avg_circle_radius ** 2
    circle_count = round(np.sum(tresholded / 255) / approx_area)
    return circle_count


def draw_hough(draw_source, ret_circles=False, draw_circles=False):
    """
    Looks for circles in image, draws them and returns average radius and
    If ret_circles == True, returns list of centers and radiuses of each circle
    Args:
        draw_source: np.array
        ret_circles: boolean
        draw_circles: boolean
    """
    img = cv2.cvtColor(draw_source, cv2.COLOR_RGB2GRAY)
    blured = cv2.medianBlur(img, 5)
    tresholded = np.invert(cv2.threshold(
                           blured, 127, 255, cv2.THRESH_BINARY)[1])

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
    circled = draw_source.copy()
    avg_r = 0
    for i in circles[0, :]:
        avg_r += i[2]
        if draw_circles:
            cv2.circle(circled, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(circled, (i[0], i[1]), 2, (255, 0, 0), 1)

    if not ret_circles:
        return circled, avg_r / len(circles[0])
    return circled, avg_r / len(circles[0]), circles[0]


AVG_H = 1381
AVG_W = 2272


def get_granule_count(processed_img, ret_dist=False):
    """
    Returns approximate count of circles in image,
    if ret_dist == True, return distribution of circles' radiuses
    """
    proc = cut_image(
        processed_img,
        50,
        50,
        50,
        50,
    )
    im_circled, avg_r, circles = draw_hough(proc, ret_circles=True)
    found = count_circles(proc, avg_r)

    resized = cv2.resize(proc, (AVG_W, AVG_H))
    _, _, circles = draw_hough(resized, ret_circles=True)
    radiuses = np.array([circle[2] for circle in circles])

    distros = np.array(
        [np.where(radiuses == i)[0].shape[0] for i in range(1, 30)])

    if ret_dist:
        return found, distros / np.sum(distros)
    return found
