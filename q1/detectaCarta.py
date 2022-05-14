import cv2
import numpy as np

template = cv2.imread("carta7.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

cap = cv2.VideoCapture("q1.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:

    ret, frame = cap.read()
    resize = cv2.resize(frame, (1400, 800))

    gray_frame = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)


    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.7)

    lower_red = np.array([0, 66, 134])
    upper_red = np.array([180, 255, 243])


    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.erode(mask, np.ones((9, 10), np.uint8))

    contours, ret = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        x = approx.ravel()[0]

        y = approx.ravel()[1]

        cv2.drawContours(resize, [approx], 0, (10, 205, 15), 5)


    for pt in zip(*loc[::-1]):
        print("carta encontrada")        

        cv2.putText(resize, 'CARTA ENCONTRADA', (40, 50), cv2.QT_FONT_BLACK,
                    1, (11, 11, 10), 2, cv2.LINE_AA)

    cv2.imshow("FindCards", resize)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()