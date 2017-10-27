import cv2
import os
#import matplotlib.pyplot as plt  

# rectangle color and stroke
color = (0,0,255)       # reverse of RGB (B,G,R) - weird
strokeWeight = 1        # thickness of outline

# set window name
windowName = "Object Detection"
i = 0

# load an image to search for faces
img = cv2.imread("3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# load detection file (various files for different views and uses)
facecascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")



# preprocessing, as suggested by: http://www.bytefish.de/wiki/opencv/object_detection

# detect objects, return as list
faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))


    # get a list of rectangles
for x,y, width,height in faces:
    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)

# save 
_path = os.getcwd()
save_path = os.path.join(_path , "selfphotos")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

if not faces is (): 
        for x,y,z,w in faces:
            roiImg = img[y:y+w,x:x+z]
            cv2.imwrite(save_path+'/' + str(i)+'.jpg',roiImg)
            cv2.rectangle(img,(x,y),(x+z,y+w),(0,0,255),2)
            i +=1

# display!
cv2.destroyAllWindows()
cv2.imshow(windowName, img)