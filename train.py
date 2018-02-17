#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import numpy as np

import os
import theano
import glob

import cv2
import matplotlib
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.externals import joblib

# Podaci za treniranje

# dimenzije slike
imageRows, imageCols = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
imageChannels = 1


# Batch_size za train
batch_size = 32

## Broj klasa( slova abecede ->26 + razmak + nista)
numberOfClasses = 28

# Broj epoha za train
numberOfEpochs = 15

# Total number of convolutional filters to use
numberOfFilters = 32
# Max pooling
pool = 2
# Size of convolution kernel
conv = 3

skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

originalPath = "dataOriginal/"    # direktorijum gde su sacuvane originalne fotografije
skinMaskPath = "dataSkinMask/"    # direktorijum gde ce se cuvati slike koje ce biti ulazi u mrezu

def loadCNN():
    '''

    Funkcija koja sluzi za ucitavanje modela
    CNN prima veliku sliku u matricnom obliku i smanjuje je primenjujuci operacije konvolucije i dropouta
    i filtera i nakon toga uproscenu sliku sa bitnim karakteristikama pretvara u niz i pusta na obicnu neuronsku mrezu

    :return: model
    '''
    global get_output
    model = Sequential()

    model.add(Conv2D(numberOfFilters, (conv, conv),
                     padding='valid',
                     input_shape=(imageChannels, imageRows, imageCols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(numberOfFilters, (conv, conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.5))

    #pretvara u niz
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses))
    model.add(Activation('softmax'))


    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model

def trainModel(model):
    '''

    Funkcija sluzi za treniranje modela

    :param model: ucitani model
    '''

    # Razdvajanje X and y u training and testing setove
    X_train, X_test, Y_train, Y_test = prepare_data()

    print("krecem")
    # Pocinje treniranje ucitanog modela
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=numberOfEpochs,
                 verbose=1, validation_split=0.125)

    visualizeHis(hist)

    print "*******************"
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    model.save("myModelForSigns.hdf5")


def visualizeHis(hist):
    '''

    Funkcija koja sluzi za prikazivanje dijagrama uspesnosti i gubitaka trening i validacionih podataka

    :param hist: istrenirani model
    '''
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(numberOfEpochs)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

def obradiSlike(imlist):
    '''

    Funkcija koja sluzi za obradu slika u boji, dodaje im se skin maska i slike se cuvaju

    :param imlist: lista imena slika u boji
    '''
    index = 0
    print("ulaziiim")
    for img in glob.glob("dataOriginal3/*.png"):
        roi = cv2.imread(img)
        # HSV values
        low_range = np.array([0, 10, 60])
        upper_range = np.array([20, 150, 255])

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Apply skin color range
        mask = cv2.inRange(hsv, low_range, upper_range)

        mask = cv2.erode(mask, skinkernel, iterations=1)
        mask = cv2.dilate(mask, skinkernel, iterations=1)

        # blur
        mask = cv2.GaussianBlur(mask, (15, 15), 1)

        # bitwise and mask original frame
        res = cv2.bitwise_and(roi, roi, mask=mask)
        # color to grayscale
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        print(imlist[index])
        cv2.imwrite('dataSkinMask3/' + imlist[index], res)
        index+=1


def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

def initializers():
    '''

    Funkcija koja sluzi za obradu slika u boji

    :return:
    '''
    imlist = modlistdir('./dataOriginal3/')

    obradiSlike(imlist)

    image1 = np.array(Image.open('./dataSkinMask3'+ '/' + imlist[0]))  # open one image to get size
    # plt.imshow(im1)


    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images

    # kreira matrix od slika koje su pretvorene u nizove
    immatrix = np.array([np.array(Image.open('./dataSkinMask3' + '/' + images).convert('L')).flatten()
                         for images in imlist], dtype='f')

    print immatrix.shape

    raw_input("Press any key")

    # sve jedinice
    label = np.ones((total_images,), dtype=int)

    samples_per_class = total_images / numberOfClasses
    print "uzoraka po klasi : ", samples_per_class
    s = 0
    r = samples_per_class

    # kreira labele
    for classIndex in range(numberOfClasses):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    # mesa zbog boljeg treniranja
    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Razdvaja X i Y, trening set je 20%
    # X su vrednosti koje dovodim na ulaz u neuronsku mrezu, a y su labele( od 0 do 300 za prvo slovo, itd....)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


    X_train = X_train.reshape(X_train.shape[0], imageChannels, imageRows, imageCols)
    X_test = X_test.reshape(X_test.shape[0], imageChannels, imageRows, imageCols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizovanje
    X_train /= 255
    X_test /= 255


    Y_train = np_utils.to_categorical(y_train, numberOfClasses)
    Y_test = np_utils.to_categorical(y_test, numberOfClasses)
    return X_train, X_test, Y_train, Y_test


def main():
    model = loadCNN()

    trainModel(model)

def prepare_data():
    '''

    Funkcija koja sluzi za pripremu podataka za treniranje modela

    :return:
    '''
    if not os.path.exists("X_train.features"):
        X_train, X_test, Y_train, Y_test = initializers()
        joblib.dump(X_train, "X_train.features")
        joblib.dump(X_test, "X_test.features")
        joblib.dump(Y_train, "Y_train.labels")
        joblib.dump(Y_test, "Y_test.labels")
    else:
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

if __name__ == "__main__":
    main()
    test()




