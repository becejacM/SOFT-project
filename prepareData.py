#-*- coding: utf-8 -*-

import cv2
import numpy as np
import time

#kernel za obradu slike
skinKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# parametri potrebni za cuvanje slika
imageSave = False
numberOfPictures = 301
nameOfSign = ""
originalPath = "dataOriginal/"    # direktorijum gde ce se cuvati snimljene fotografije
skinMaskPath = "dataSkinMask/"
count = 0

# parametri ekrana u boji
x = 400
y = 200
h = 200
w = 200

# parametri za ispis teksta na ekranu u boji
font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
fx = 10
fy = 355
fh = 18


def saveImage(image):
    '''

    Funkcija koja vrsi cuvanje slika u boji

    :param image: slika u boji

    '''
    global count, nameOfSign, imageSave
    if count > (numberOfPictures - 1):
        # Reset the parameters
        imageSave = False
        nameOfSign = ''
        count = 0
        return



    count = count + 1
    name = nameOfSign + str(count)
    print("Saving image:", name)
    cv2.imwrite(originalPath + name + ".png", image)
    time.sleep(0.04)


def skinMask(frame, x, y, w, h):
    '''

    Postavljanje skin maske na sliku u pomocnom prozoru da bi se videlo kako
    ce izgledati slika kada se preradi i posalje mrezi za obucavanje

    :param frame: slika u boji
    :param x0: x-osa
    :param y0: y-osa
    :param width: sirina slike
    :param height: visina slike
    :return: slika kakva ce biti kasnije poslata neuronskoj mrezi

    '''
    low_range = np.array([0, 10, 60])
    upper_range = np.array([20, 155, 255])

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(frame, '|', (x+h/3, y+w), font, size, (0, 255, 0), 1, 1)
    cv2.putText(frame, '|', (x+h/3*2, y+w), font, size, (0, 255, 0), 1, 1)

    roi = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Dodavanje skin color range-a
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinKernel, iterations=1)
    mask = cv2.dilate(mask, skinKernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    # bitwise i mask originalne slike
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color => grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res

def main():

    global x, y, w, h, imageSave, nameOfSign

    # otvori kameru
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)


    while (True):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 3)
        if ret == True:
            roi = skinMask(frame, x, y, w, h)

        if imageSave == True:
            savedFrame = frame[y:y + h, x:x + w]
            saveImage(savedFrame, roi)

        cv2.putText(frame, 'Options:', (fx, fy), font, 0.7, (0, 255, 0), 2, 1)
        cv2.putText(frame, 'Press n to enter the name of sign', (fx, fy + 2 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'Press s to start capturing', (fx, fy + 3 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'Press ESC to exit', (fx, fy + 6 * fh), font, size, (0, 255, 0), 1, 1)

        cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        if key == 27:
            break

        if key == ord('s'):
            imageSave = not imageSave

            if nameOfSign != '':
                imageSave = True
            else:
                print "Enter a name of sign first, by pressing 'n'"
                imageSave = False

        elif key == ord('n'):
            nameOfSign = raw_input("Enter the name of sign: ")

    # Oslobodi i unisti sve prozore
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()