import cv2
import numpy as np


def region_selection(
    image, bottom_left_rate, top_left_rate, bottom_right_rate, top_right_rate
):
    """
    This function is to focus region of interest, instead of all area of image
    ROI color is white, other is black
    """
    mask = np.zeros_like(image)
    ignore_mask_color = 255

    # Creating a polygon to focus only on the round in the picture
    row, col = image.shape[:2]
    bottom_left = [col * bottom_left_rate[0], row * bottom_left_rate[1]]
    top_left = [col * top_left_rate[0], row * top_left_rate[1]]
    bottom_right = [col * bottom_right_rate[0], row * bottom_right_rate[1]]
    top_right = [col * top_right_rate[0], row * top_right_rate[1]]

    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32
    )

    # filling the polygon with white color and generate the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # Performing Bitwise AND on the input image and mask to get only the edge ROI
    mask_image = cv2.bitwise_and(image, mask)

    return mask_image


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
            lines: output from Hough Transform
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    #
    left_lane = (
        np.dot(left_weights, left_lines) / np.sum(left_weights)
        if len(left_weights) > 0
        else None
    )
    right_lane = (
        np.dot(right_weights, right_lines) / np.sum(right_weights)
        if len(right_weights) > 0
        else None
    )
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
            Parameters:
                    image: The input test image.
                    lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line


def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
    """
    Draw lines onto the input image.
            Parameters:
                    image: The input test image (video frame in our case).
                    lines: The output lines from Hough Transform.
                    color (Default = red): Line color.
                    thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
