from imutils.video import VideoStream
import pyrealsense2 as rs
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
from sklearn.metrics import accuracy_score
import numpy as np
from time import time

import cv2
pipe= rs.pipeline()
cfg= rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
pipe.start(cfg)


# initialize 'currentname' to trigger only when a new person is identified
currentname = "Paras"
# determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings_gruhit.pickle"
# use this xml file
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=1).start()
#vs = VideoStream(usePiCamera=True).start()


# start the FPS counter

l=[]
l_time=[]

time1 = 0

# loop over frames from the video file stream
while(True):
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	
	frame1 = pipe.wait_for_frames()
	depth_frame = frame1.get_depth_frame()
	depth_image = np.asanyarray(depth_frame.get_data())
	color_frame = frame1.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())
	depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha = 0.5), cv2.COLORMAP_JET)
	
	#color_image = np.asanyarray(vs.read())



	if color_image is not None:
		
		color_image = cv2.resize(color_image, (0, 0), fx=0.5, fy=0.5)
	
	   # convert the input frame from (1) BGR to grayscale (for face
	  # detection) and (2) from BGR to RGB (for face recognition)
		gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

		# detect faces in the grayscale frame
		rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		# OpenCV returns bounding box coordinates in (x, y, w, h) order
		# but we need them in (top, right, bottom, left) order, so we
		# need to do a bit of reordering
		boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
		

		# compute the facial embeddings for each face bounding box
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []
		

		# loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding,tolerance=0.415)
			name = "Unknown" # if face is not recognized, then print Unknown

			# check to see if we have found a match
			if True in matches:
				
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
			

				# loop over the matched indexes and maintain a count for
				# each recognized face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1
					

				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
				name = max(counts, key=counts.get)
				
				# if someone in your dataset is identified, print their name on the screen
				if currentname != name:
					currentname = name
					print(currentname)
					l.append(currentname)
					
					
					
					
					
			
			# update the list of names
			names.append(name)
		
		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
			# draw the predicted face name on the image - color is in BGR
			cv2.rectangle(color_image, (left, top), (right, bottom),
				(0, 255, 225), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(color_image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				.8, (0, 255, 255), 2)
		time2 = time()
		if (time2 - time1) > 0 :
        # Calculate the number of frames per second.
			frames_per_second = 1.0 / (time2 - time1)
            # Write the calculated number of frames per second on the frame. 
			cv2.putText(color_image, 'FPS: {}'.format(int(frames_per_second)), (10, 
            30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            # Update the previous frame time to this frame time.
            # As this frame will become previous frame in next iteration.
			time1 = time2	

		# display the image to our screen
		cv2.imshow("Facial Recognition is Running", color_image)
		cv2.imshow('depth', depth_cm)

		frame=color_image
		
		key = cv2.waitKey(1) & 0xFF
		
		# quit when 'q' key is pressed

		if key == ord("q"):
			break

		# update the FPS counter
	
		
        # stop the timer and display FPS information 
	else:
		
		break








#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
pipe.stop()
cv2.destroyAllWindows()


