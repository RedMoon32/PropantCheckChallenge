import numpy as np
import cv2


def apply_filter(image:np.array) -> np.array:
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array
    Returns:
        filtered: np.array
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float64) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    return filtered


def apply_threshold(filtered:np.array) -> np.array:
    """
    Apply OTSU threshold
    Args:
        filtered: np.array
    Returns:
        thresh: np.array
    """
    thresh = cv2.threshold(filtered, 127, 255,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh


def cut_image(
    img:np.array,
    padding:int
) -> np.array:
    """
    Center crop of image with given boundaries
    Args:
        img: np.array
        padding: int
    Returns:
        img: np.array
    """
    return img[
        padding: img.shape[0] - padding,
        padding: img.shape[1] - padding,
    ]


def cut_borders(img:np.array) -> np.array:
    """
    Automaticly crop black rectangle in the photo
    Args:
        img: np.array - initial image
    Returns:
        img: np.array - cropped image
    """
    x_left = 0
    x_right = img.shape[0] - 1
    y_left = 0
    y_right = img.shape[1] - 1

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = apply_threshold(gray)

    while thresh[x_left][y_left] == 255:
        x_left += 1
        y_left += 1

    while thresh[x_right][y_right] == 255:
        x_right -= 1
        y_right -= 1

    while thresh[x_left][y_left] == 0:
        x_left += 1
        y_left += 1

    while thresh[x_right][y_right] == 0:
        x_right -= 1
        y_right -= 1

    return img[x_left:x_right, y_left:y_right]


def detect_contour(img:np.array, image_shape:tuple) -> (np.array, list):
    """
    Detect image contours and draw them on empty image with given shape
    Returns new image and list of contours
    Args:
        img: np.array()
        image_shape: tuple
    Returns:
        canvas: np.array() - empty image with contours drawn
        cnt: list - list of contours
    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # second biggest contour is our goal rectangle
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)

    return canvas, cnt


def detect_corners_from_contour(canvas:np.array, cnt:np.array) -> list:
    """
    Detecting corner points from contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list
    Returns:
        approx_corners: list
    """
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())

    return approx_corners


def order_points(pts:list) -> list:
    """
    Rearrange the corner points so that first entry is top-left,
    second is top-right, third is bottom-right, fourth is bottom-left.
    Args:
        pts: list - corner points 
    Returns:
        rect: list - rearranged points
    """
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image:np.array, pts:list) -> np.array:
    """
    Apply perspective transform to image
    Args:
        image: np.array - initial image
        pts: list - corner points of rectangle
    Returns:
        warped: np.array - perspective transformed image
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def automatic_brightness_and_contrast(image:np.array, clip_hist_percent:float = 1) -> (np.array, float, float):
    """
    Brightness and contrast normalization of the image
    by clipping grays in histogram
    Args:
        image: np.array
        clip_hist_percent: float
    Returns:
        auto_result: np.array - balanced image
        alpha: float - alpha used in cv2.convertScaleAbs
        beta: float - beta used in cv2.convertScaleAbs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def full_pipeline(img:np.array) -> np.array:
    """
    Full preprocssing of the image
    Args:
        img: np.array - Initial Image
    Returns:
        img: np.array - cropped by rectangle image
    """
    # rawly cut image so it does find contour of the paper itself
    cutted_img = cut_image(img, 500)
    filtered = apply_filter(cutted_img)
    thresholded = apply_threshold(filtered)
    canvas, contour = detect_contour(thresholded, thresholded.shape)
    corners = detect_corners_from_contour(canvas, contour)
    cropped = four_point_transform(cutted_img, np.array(corners))
    # cut black borders
    cutted = cut_borders(cropped)
    # improve brightness
    res = automatic_brightness_and_contrast(cutted, clip_hist_percent=1)[0]
    img = np.rot90(res) if img.shape[0] > img.shape[1] else res
    return img
