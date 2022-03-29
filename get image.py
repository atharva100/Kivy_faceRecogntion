import cv2
import os
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        #cv2.rectangle(frames, (x, y), (x + w, y + h),  (0, 0, 255), 2)
        faces = frames[y:y + h, x:x + w]
        cv2.imshow("face", faces)
        cv2.imwrite('face.jpg', faces)   # save image
        cv2.waitKey(0)

cv2.destroyAllWindows()
