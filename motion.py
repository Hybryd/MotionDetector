import cv2
import datetime
import imutils
import numpy as np
import time

# Read the camera parameters
camera_conf = open('camera.conf', 'r')
lines = camera_conf.readlines()

ip          = lines[0].replace("\n","")
login       = lines[1].replace("\n","")
password    = lines[2].replace("\n","")

camera_address = "rtsp://"+login+":"+password+"@"+ip

# Open camera stream
capture = cv2.VideoCapture(camera_address)

thresh_motion = 3000
fps = 50 # There are still problems with the speed of the recorded video

kernel = np.ones((5, 5))

previous_frame = None
presence = False
record = False
writer = None

patience = 50
patience_cpt = 0
display_contours = True

show_image = True

while(True):
    ret, frame = capture.read()
	
    if frame is None:
        break

    # Preprocessing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    
    if previous_frame is None:
        previous_frame = frame_gray
        continue

    delta = cv2.absdiff(previous_frame, frame_gray)
    previous_frame = frame_gray
    thresh = cv2.dilate(delta, kernel, iterations=1)
    thresh = cv2.threshold(thresh, 20, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Keep only the contours that are big enough
    real_contours = [c for c in contours if cv2.contourArea(c) > thresh_motion]

    # If there is at least 1 detected contour
    if len(real_contours) > 0 :
        presence = True
        patience_cpt = 0

        if display_contours:
            for c in real_contours:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Start recording in a new video file
        if presence and not record :
            record = True
            writer = cv2.VideoWriter(   datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")+'.mp4',
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        20,
                                        (int(capture.get(3)), int(capture.get(4)))
                                    )
        
        if writer != None:
            writer.write(frame)        
    
    # If nothing is detected
    else:
        # Wait few frames before ending the recording
        if patience_cpt < patience :
            patience_cpt += 1
            if writer != None:
                writer.write(frame)  
        
        # End the recording
        else :
            record = False
            presence = False

            if writer != None :
                writer.release()
            writer = None
            patience_cpt = 0
    
    if show_image :
        height, width, layers = frame.shape
        new_h = height // 2
        new_w = width // 2
        frame_resized = cv2.resize(frame, (new_w, new_h))
        cv2.imshow('Camera', frame_resized)

    cv2.waitKey(1)

capture.release()
if writer != None :
    writer.release()

cv2.destroyAllWindows()
