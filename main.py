import json

import cv2
import numpy as np

from utils import draw_lane_lines, lane_lines, region_selection

# Read config data from json file
with open("config/config.json") as config_json_file:
    data = json.load(config_json_file)

# Defining constant value:
bottom_left_rate = data["bottom_left_rate"]
top_left_rate = data["top_left_rate"]
top_right_rate = data["top_right_rate"]
bottom_right_rate = data["bottom_right_rate"]

# Read Video from video capture
cap = cv2.VideoCapture(data["video_path"])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # display original frame
    # cv2.imshow("Origin", frame)

    # convert to grayscale frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayScale", gray_frame)

    # Noise filter, using Gaussian Noise
    gaussian_blur_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # cv2.imshow("Gaussian filter", gaussian_blur_frame)

    # Edge detection, using Cany Algorithm
    edge = cv2.Canny(gray_frame, 50, 200)
    # cv2.imshow("Edge", edge)

    # ROI region of interest - Ego lane
    ROI = region_selection(
        edge, bottom_left_rate, top_left_rate, bottom_right_rate, top_right_rate
    )
    # cv2.imshow("ROI", ROI)

    # Line detection
    # Apply Hough Lines P method to directly obtain line and points
    lines = cv2.HoughLinesP(
        image=ROI,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=5,
        maxLineGap=10,
    )
    # # Draw the left and right lane to the screen
    if lines is None:
        result = frame
    else:
        result = draw_lane_lines(frame, lane_lines(frame, lines))
    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
