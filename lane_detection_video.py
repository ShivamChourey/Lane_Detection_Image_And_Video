import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/3))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

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


def Canny(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return canny

def region_of_interest(im):
    height = im.shape[0]
    polygons = np.array([
    [(200,height),(1100, height),(550,250)]
    ])
    mask = np.zeros_like(im)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_lane = np.bitwise_and(im, mask)
    return masked_lane

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)
    return line_image


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = Canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    averaged_lines = average_lines(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # cv2.imshow("Result", lane)
    cv2.imshow("Detected_Lanes", combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
