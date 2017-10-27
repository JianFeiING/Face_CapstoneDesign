#
import cv2
import sys 
import os
# import matplotlib.pyplot as plt

# Get user supplied values
__Path = os.getcwd()
imagePath = __Path + "/4.jpg"
cascPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60,60),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# rectangle color and stroke
color = (0,0,255)       # reverse of RGB (B,G,R) - weird
strokeWeight = 1        # thickness of outline

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
     cv2.rectangle(image, (x,y), (x+w, y+h), color, strokeWeight)

# save the photo
save_path = os.path.join(__Path , "selfphotos")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

i = 0
if not faces is (): 
        for x,y,z,w in faces:
            roiImg = image[y:y+w,x:x+z]
            cv2.imwrite(save_path+'/' + str(i)+'.jpg',roiImg)
            cv2.rectangle(image,(x,y),(x+z,y+w),(0,0,255),2)
            i +=1

# display
cv2.destroyAllWindows()
cv2.imshow("Faces found", image)
