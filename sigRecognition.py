#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import cv2
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import numpy as np
import time

# dimenzije slike
imageRows, imageCols = 200, 200

guessSign = False

lastgesture = -1

# broj kanala
# za grayscale ide 1, za kolor bi bilo 3 zbog RGB
imageChannels = 1

# Batch_size za trening
batch_size = 32

## BRoj klasa
numberOfClasses = 28

# Broj convolutional filters
numberOfFilters = 32
# Max pooling
pool = 2
# Velicina convolution kernel-a
conv = 3

font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
fx = 10
fy = 355
fh = 18

x0 = 400
y0 = 200
height = 200
width = 200
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
output = ["A","B","C","D","E","F","G","H","I", "J","K","L","M","N","NOTHING","O","P","Q","R","SPACE","S","T","U","V","W","X","Y","Z"]

sentence=""
nextLetter=""
saveLetter=False
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

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.summary()
    model.get_config()


    fname = "myModelForSigns.hdf5"
    print ("ucitavam ", fname)
    model.load_weights(fname)

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model

def skinMask(frame, x0, y0, width, height):
    global guessSign, mod, lastgesture, saveLetter, sentence, nextLetter
    # HSV values
    low_range = np.array([0, 10, 60])
    upper_range = np.array([20, 150, 255])

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    cv2.putText(frame, '|', (x0+height/3, y0+width), font, size, (0, 255, 0), 1, 1)
    cv2.putText(frame, '|', (x0+height/3*2, y0+width), font, size, (0, 255, 0), 1, 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

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

    if guessSign == True:
        retgesture = guessSignFunction(mod, res)
        #print "ffffffffffffffffff "+retgesture
        if retgesture==155:
            retgesture=lastgesture
        if output[retgesture] != 'NOTHING' and retgesture!=lastgesture:
            lastgesture = retgesture
            print output[lastgesture]
            nextLetter = output[lastgesture]
            #cv2.putText(frame, 'Sign:  ' + output[lastgesture], (fx, fy + 4 * fh), font, size, (0, 255, 0), 1, 1)
            time.sleep(0.1)
        if saveLetter==True:
            if nextLetter=="SPACE":
                sentence+=" "
            else:
                sentence += nextLetter
            saveLetter = False

    return res

def guessSignFunction(model, img):
    global output, get_output
    # Ucitavanje slike

    image = np.array(img).flatten()

    # risejpovanje
    image = image.reshape(imageChannels, imageRows, imageCols)

    # float32
    image = image.astype('float32')

    # normalizovanje
    image = image / 255

    # risejp za nn
    rimage = image.reshape(1, imageChannels, imageRows, imageCols)

    prob_array = get_output([rimage, 0])[0]

    #print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.iteritems(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 70.0:
        #print guess + "  Probability: ", prob

        return output.index(guess)

    else:
        return 155


def Main():
    global guessSign, mod, x0, y0, width, height, saveLetter, sentence, nextLetter


    print "Krecem sa ucitavanjem istreniranog modela"
    mod = loadCNN()

    # otvori kameru
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('second_quote.avi', fourcc, 20.0, (640, 480))

     # set rt size as 640x480
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    while (True):
        ret, frame = cap.read()
        max_area = 0

        frame = cv2.flip(frame, 3)

        if ret == True:
            roi = skinMask(frame, x0, y0, width, height)

        cv2.putText(frame, 'Options:', (fx, fy), font, size, (255,105,180), 2, 1)
        cv2.putText(frame, 'g - Toggle Prediction Mode', (fx, fy + 2 * fh), font, size, (255,105,180), 1, 1)
        cv2.putText(frame, 's - Save letter if you are ready', (fx, fy + 4 * fh), font, size, (255,105,180), 1, 1)
        cv2.putText(frame, 'Sentence:  ' + sentence, (fx+10, 30), font, 0.7, (0,0,255), 2, 1)
        cv2.putText(frame, 'ESC - Exit', (fx, fy + 6 * fh), font, size, (255,105,180), 1, 1)

        cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        ## Izlaz iz programa na ESC
        if key == 27:
            break

        ## kad se pritisne g pokrece se i zaustavlja predvidjanje
        elif key == ord('g'):
            guessSign = not guessSign
            print "Rezim predvidjanja je : {}".format(guessSign)
        elif key == ord('s'):
            saveLetter = True
            print "Cuvam : {}".format(saveLetter)
        out.write(frame)

    # Oslobodi i unisti sve prozore
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()