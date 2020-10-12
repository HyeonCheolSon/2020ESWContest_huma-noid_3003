# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\svh\AppData\Local\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS, 10)


def color_img(img, push_color):
    if push_color == "red":

        lower = (0 - 10, 60, 60)

        upper = (0 + 10, 255, 255)



    elif push_color == "green":

        lower = (60 - 10, 100, 100)

        upper = (60 + 10, 255, 255)



    elif push_color == "yellow":

        lower = (30 - 10, 100, 100)

        upper = (30 + 10, 255, 255)



    elif push_color == "blue":

        lower = (120 - 20, 60, 60)

        upper = (120 + 20, 255, 255)

    img_color = img

    height, width = img_color.shape[:2]

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    img_mask = cv2.inRange(img_hsv, lower, upper)

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    return img_result


if cap.isOpened():

    while True:

        ret, fram = cap.read()

        if ret:

            img_color = color_img(fram, 'blue')

            copy = img_color.copy()

            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            ALPHABET_number = 0

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                ALPHABET = img_color[y-10:y + h+10, x-10:x + w+10]
                ALPHABET = cv2.resize(ALPHABET, dsize=(100, 100), interpolation=cv2.INTER_AREA)
                ALPHABET = cv2.cvtColor(ALPHABET, cv2.COLOR_HSV2BGR)
                ALPHABET = cv2.cvtColor(ALPHABET, cv2.COLOR_BGR2GRAY)
                ALPHABET = cv2.threshold(ALPHABET, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
                #ret1, ALPHABET = cv2.threshold(ALPHABET, 5, 255, cv2.THRESH_BINARY)
                #ALPHABET = cv2.pyrDown(ALPHABET)
                #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                kernel = np.ones((5, 5), np.uint8)
                ALPHABET = cv2.morphologyEx(ALPHABET, cv2.MORPH_CLOSE, kernel)
                ALPHABET = cv2.morphologyEx(ALPHABET, cv2.MORPH_OPEN, kernel)
                # cv2.imwrite('ALPHABET_{}.png'.format(ALPHABET_number), ALPHABET)

                cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)

                ALPHABET_number += 1

                alpha = pytesseract.image_to_string(ALPHABET, lang="Eng")

                if (ord(alpha) >= 65 & ord(alpha) <= 68) :
                    print(alpha)

                cv2.imshow('final', ALPHABET)

            cv2.imshow('copy', copy)

            cv2.imshow('img_result', img_color)

            cv2.imshow("camera", fram)

            if cv2.waitKey(1) != -1:
                cv2.imwrite("../photo.jpg", fram)

                break

        else:

            print("no fram")

            break

else:

    print("can't open camera")

cap.release()

cv2.destroyAllWindows()
