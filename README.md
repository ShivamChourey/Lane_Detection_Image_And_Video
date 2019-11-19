# Lane_Detection_Image_And_Video
This repository contains a two Python-OpenCV files. The file "lane_detetcion_image.py" detected lane in an image, while the file "lane_detection_video.py" detects lanes in a video.


The technique used for lane-detection involves finding edges in an image or video-frame using Canny edge detection, then applying Linear Hough transform to identify the dominant lines in a region of image, and then averaging the detected lines to identify lanes. The detected lanes are then displayed on the image/frame.

The video used in the code is available at https://github.com/rslim087a/road-video 

I learned this lesson thanks to tutorial created by Rayan Slim and the ProgrammingKnowledge.
