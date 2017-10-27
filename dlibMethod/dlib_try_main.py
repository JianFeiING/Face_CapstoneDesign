#
#
import sys
import os
import cv2
import dlib
import glob
from skimage import io
#
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#win = dlib.image_window()

color = (0,255,0)       # parameters for the box::color
strokeWeight = 3        # parameters for the box::thickness of outline

img = io.imread("15.png")
img = img[:, :, :3].copy()

# for f in sys.argv[1:]:
#     print("Processing file: {}".format(f))
#     img = io.imread(f)

# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
i = 0
font = cv2.FONT_HERSHEY_DUPLEX
name = ['Tian', 'Haoyang','Weizhao', 'Teer', 'Shikun']
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom() + 30),(0,0,255),2)
    cv2.rectangle(img,(d.left(), d.bottom() - 5), (d.right(), d.bottom() + 30), (0,0,255), -1)

    cv2.putText(img, name[i], (d.left() + 6, d.bottom() + 24), font, 1.0, (255,255,255), 1)
    i = i + 1


io.imsave("newImg1.png",img)
print("image with boxed face saved!")
        
# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.

# if (len(sys.argv[1:]) > 0):
#     img = io.imread(sys.argv[1])
dets, scores, idx = detector.run(img, 1, -1)
for i, d in enumerate(dets):
    print("Detection {}, score: {}, face_type:{}".format(
        d, scores[i], idx[i]))

