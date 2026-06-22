from skimage.filters import threshold_local
from transform import four_point_transform
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height=500)

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
edges = cv2.Canny(grayscale, 75, 200)

print("step 1: edge detection")
cv2.imshow("original image", image)
cv2.imshow("after edge detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

document_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        document_contour = approx
        break

print("step 2: contour detection")
if document_contour is None:
    print("no document contour found")
else:
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
    cv2.imshow("after contour detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    top_down = four_point_transform(original, document_contour.reshape(4, 2) * ratio)

    top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    threshold_map = threshold_local(top_down, 11, offset=10, method="gaussian")
    top_down = (top_down > threshold_map).astype("uint8") * 255

    print("step 3: perspective transform")
    cv2.imshow("original", imutils.resize(original, height=650))
    cv2.imshow("scanned", imutils.resize(top_down, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
