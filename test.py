#-*- coding: utf-8 -*-

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import numpy as np
import glob

from sklearn.externals import joblib

from sigRecognition import *

def prepare_data():
    '''

    Funkcija koja sluzi za pripremu podataka za treniranje modela

    :return:
    '''
    X_train = joblib.load("X_train.features")
    X_test = joblib.load("X_test.features")
    Y_train = joblib.load("Y_train.labels")
    Y_test = joblib.load("Y_test.labels")
    return X_train, X_test, Y_train, Y_test

def test():
    '''

    Funkcija koja sluzi za proveru uspesnosti modela

    '''
    X_train, X_test, Y_train, Y_test = prepare_data()
    model = load_model("myModelForSigns.hdf5")
    pred_classes = model.predict_classes(X_test)
    test_class = [np.argmax(label) for label in Y_test]
    matches = pred_classes == test_class
    num_of_matches = sum(matches)
    success = float(num_of_matches) / len(matches)
    print("Number: ",num_of_matches)
    print("Matches: ",len(matches))
    print("Success: %f", success * 100)


def recognize(img, mod):
    roi = cv2.imread(img)
    low_range = np.array([0, 10, 60])
    upper_range = np.array([20, 150, 255])

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    # cv2.imshow("Blur", mask)

    res = cv2.bitwise_and(roi, roi, mask=mask)
    # konvertovanje u grejskal
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    retgesture = guessSignFunction(mod, res)
    return output[retgesture]

def test1():
    ''' HELLO WORLD'''
    sentence = ['H', 'E', 'L', 'L', 'O', ' ', 'W', 'O', 'R', 'L', 'D']
    words = ["HELLO", "WORLD"]
    file = "tests/test1/*.png"
    testSignRecognition(file,sentence,words)

def test2():
    ''' I AM MILANA'''
    sentence = ['I', ' ', 'A', 'M', ' ', 'M', 'I', 'L', 'A', 'N', 'A']
    words = ["I", "AM", "MILANA"]
    file = "tests/test2/*.png"
    testSignRecognition(file,sentence,words)

def test3():
    ''' I LOVE YOU '''
    sentence = ['I', ' ', 'L', 'O', 'V', 'E', ' ', 'Y', 'O', 'U']
    words = ["I", "LOVE", "YOU"]
    file = "tests/test3/*.png"
    testSignRecognition(file,sentence,words)

def testSignRecognition(file, sentence, reci):
    model = loadCNN()
    recSentence=""
    text=[]
    trRec=""
    for img in glob.glob(file):
        r = recognize(img, model)
        if r=="SPACE":
            recSentence += " "
            text.append(trRec)
            trRec=""
        else:
            recSentence+= r
            trRec+=r
    text.append(trRec)


    brPogodjenih = 0.00
    index = 0
    print len(sentence)
    print (recSentence)
    for i in recSentence:
        if i == sentence[index]:
            brPogodjenih += 1
        index += 1
    numSentence = len(sentence)
    procenat = (brPogodjenih / numSentence) * 100.00
    print "**********************************"
    print "Broj i procenat pogodjenih slova:"
    print "Ukupno pogodjenih: %.0f " % (brPogodjenih), "od: ", len(sentence)
    print "a to je: %.2f %%" % (procenat)

    print "**********************************"
    print "Broj i procenat pogodjenih reci : "
    brPogodjenihReci = 0.00
    indeks2 = 0
    for t in text:
        print t, reci[indeks2]
        if t == reci[indeks2]:
            brPogodjenihReci += 1
        indeks2 += 1
    brSlova = len(reci)
    procenatPogodjenihReci = (brPogodjenihReci / brSlova) * 100
    print "Ukupno pogodjenih reci: %.0f " % (brPogodjenihReci), "od: ", len(reci)
    print "a to je: %.2f %%" % (procenatPogodjenihReci)

if __name__ == "__main__":
    #test()
    test1()
    test2()
    test3()