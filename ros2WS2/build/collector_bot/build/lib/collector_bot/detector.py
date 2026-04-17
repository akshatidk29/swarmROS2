"""Camera-based object detection via HSV colour filtering."""

import cv2
import numpy as np

# ── HSV thresholds (forgiving for Gazebo lighting) ──
_RED_LO1 = np.array([0,   100, 80])
_RED_HI1 = np.array([10,  255, 255])
_RED_LO2 = np.array([165, 100, 80])
_RED_HI2 = np.array([180, 255, 255])

_BLUE_LO = np.array([95,  100, 80])
_BLUE_HI = np.array([135, 255, 255])

MIN_CONTOUR_AREA = 300          # px² — ignore noise


def detect_objects(cv_image):
    """
    Detect RED cubes and BLUE spheres.

    Returns
    -------
    list of (obj_type, cx, cy, area)  sorted by area descending.
    obj_type is 'cube' or 'sphere'.
    """
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask_red = (cv2.inRange(hsv, _RED_LO1, _RED_HI1) |
                cv2.inRange(hsv, _RED_LO2, _RED_HI2))
    mask_blue = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)

    results = []
    for mask, obj_type in [(mask_red, 'cube'), (mask_blue, 'sphere')]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            results.append((obj_type, cx, cy, area))

    results.sort(key=lambda r: r[3], reverse=True)
    return results
