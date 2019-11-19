# Shivam Chourey
# Lane detection on a video
# I learned this technique following tutorial by Rayan Smiles and PrgrammingKnowledge

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to get the coordinates where lines must be drawn
# The lines are given by the argument line_parameters
def make_coord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/3))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# This function takes in a set of lines, identifies the left and right lines based on slope
# Averages the left set and right set of lines to return a single left and a right line
def average_lines(image, lines):
    left = []
    right = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    left_line = make_coord(image, left_avg)
    right_line = make_coord(image, right_avg)
    return np.array([left_line, right_line])

# This function takes in an image, converts it to grayscale and performs Canny edge detection
def Canny(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return canny

# This function defines a region of interest in the image where we want to identify lanes
# Right now, this has hard-coded values specific to the image and videos used
# I plan to make it generic in near future
def region_of_interest(im):
    height = im.shape[0]
    polygons = np.array([
    [(200,height),(1100, height),(550,250)]
    ])
    mask = np.zeros_like(im)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_lane = np.bitwise_and(im, mask)
    return masked_lane

# Function  to display lines on an image, both passed as argument
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)
    return line_image

# Main program
# Capture the video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    # Read the frame
    _, frame = cap.read()

    # Detect edges
    canny_image = Canny(frame)
    # Identify the region of interest
    cropped_image = region_of_interest(canny_image)
    # Apply hough transform to identify lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    # Get a single left lane marker and right lane marker
    averaged_lines = average_lines(frame, lines)
    # Display the detected line
    line_image = display_lines(frame, averaged_lines)
    # Create a combination of original image and the detected lanes
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Display the detected lane for current frame
    cv2.imshow("Detected_Lanes", combo_image)
    # Wait 1 ms or till q is pressed on keyboard
    if cv2.waitKey(1) == ord('q'):
        break
# Release the video        
cap.release()
cv2.destroyAllWindows()
