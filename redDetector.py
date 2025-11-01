import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # converting to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define red boundaries
    lower_red1 = np.array([0, 150, 120])
    upper_red1 = np.array([8, 255, 255])

    lower_red2 = np.array([172, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    # mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # largest red object
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:

            # bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2

            # draw rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # return coordinates
            print(f"Red object detected at: x={cx}, y={cy}")

    cv2.imshow("Red Object Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
