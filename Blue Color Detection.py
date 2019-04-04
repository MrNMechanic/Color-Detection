import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(1,640)
cam.set(1,480)

while (1):
    ret, frame = cam.read()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_blue = np.array([90, 100, 140], np.uint8)
    upper_blue = np.array([120  , 250, 255], np.uint8)
    
    blue = cv2.inRange(RGB, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)

    blue = cv2.dilate(blue, kernel)
    res = cv2.bitwise_and(frame, frame, mask=blue)

    (_, cnt, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(cnt):
        area = cv2.contourArea(cnt)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(cnt)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(img, "Blue", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
