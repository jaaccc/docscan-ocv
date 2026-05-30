import cv2
import numpy as np


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    d = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]  # tl
    rect[1] = points[np.argmin(d)]  # tr
    rect[2] = points[np.argmax(s)]  # br
    rect[3] = points[np.argmax(d)]  # bl

    return rect


def four_point_transform(image, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    height = max(int(np.linalg.norm(tl - bl)), int(np.linalg.norm(tr - br)))
    width = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))

    dest = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dest)
    return cv2.warpPerspective(image, M, (width, height))
