# Make by Amirali Zandi
import cv2
import pathlib

mycascade = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(mycascade))
cam = cv2.VideoCapture(0)

while True:
    _ , frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = clf.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors = 6,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
 for (x,y,width,height) in face:
        cv2.rectangle(frame, (x,y), (x+width,y+height),(255.0,0),4)

    cv2.imshow("amirali",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()

cv2.destroyAllWindows()
