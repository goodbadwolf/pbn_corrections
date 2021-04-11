from skimage.filters import threshold_local
import argparse
import cv2
import imutils
import numpy as np
import os


# Adapted from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def is_image(path):
    possible_image = cv2.imread(path)
    success = isinstance(possible_image, np.ndarray)
    return (success, possible_image)


def detect_corners(image):
    positions = []

    def collect_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            pos = (x, y)
            positions.append(pos)
            print(f"Adding {pos}")
            if len(positions) > 4:
                positions.pop(0)

    window_name = "Input Image"

    cv2.namedWindow(window_name)

    # These two lines will force the window to be on top with focus.
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cv2.setMouseCallback(window_name, collect_mouse)
    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            break
    cv2.destroyAllWindows()
    return positions


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, nargs='+', help="Input images")
ap.add_argument("-o", "--output", required=True, help="Path to the output folder of images")
ap.add_argument("--height", required=False, help="Height to resize images before processing", default=1200)
args = vars(ap.parse_args())

for path in args["input"]:
    success, image = is_image(path)
    if success:
        print(f"Using {path}")
        image = cv2.imread(path)
        height = args["height"]
        image  = imutils.resize(image, height=height)

        corners = detect_corners(image)
        corners = np.array(corners)
        corrected_image = four_point_transform(image, corners)
        corrected_image = imutils.resize(corrected_image, height=height)
        cv2.imshow("Corrected Image", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_name = os.path.basename(path)
        write_path = os.path.join(args["output"], output_name)
        cv2.imwrite(write_path, corrected_image)
        print(f"Wrote {write_path}")
    else:
        print(f"{path} was not detected as an image file")
