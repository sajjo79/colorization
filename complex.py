from keras.layers import Input,Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from os import listdir
from os.path import isfile, join
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
#===================================
# Get images
img_rows=400
img_cols=400
class batching:
    def __init__(self,train_dir,batch_sz):
        self.train_dir=train_dir
        self.train_files = [f for f in listdir(train_dir) if (isfile(join(train_dir, f)))]
        self.img_rows=400
        self.img_cols=400
        self.batch_sz=batch_sz
        self.current_batch=0
        self.current_batch_files=[]
        self.train_batches = math.floor(len(self.train_files) / self.batch_sz)


    def DoReadImages(self,current_batch_files):
        images = []
        for i in range(len(self.current_batch_files)):
            img = img_to_array(load_img(self.train_dir + self.current_batch_files[i]))
            img = np.array(img, dtype=float)
            self.img_rows = img.shape[0]
            self.img_cols = img.shape[1]
            images.append(img)
        # ValueError: setting an array element with a sequence
        images = np.asarray(images)
        #image = img_to_array(load_img(self.Train_Dir + self.current_batch_files[i]))
        #image = np.array(image, dtype=float)
        return images

    def DoPreprocess(self,images):
        # Import map images into the lab colorspace
        img_cnt=images.shape[0]
        X_lab = rgb2lab(1.0 / 255 * images)

        #X = rgb2lab(1.0/255*image)[:,:,0]
        #Y = rgb2lab(1.0/255*image)[:,:,1:]

        X=X_lab[:,:,:,0]
        Y=X_lab[:,:,:,1:]
        Y = Y / 128
        X = X.reshape(img_cnt, img_rows, img_cols, 1)
        Y = Y.reshape(img_cnt, img_rows, img_cols, 2)
        np.save("npzs/simple_X",X)
        np.save("npzs/simple_Y", Y)
        return X,Y

    def DoModel_Sequential(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))

        # Building the neural network
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        return model

    def DoModel_Model(self):
        # Design the neural network =============================================
        inputs = Input(shape=(400, 400, 1))
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same',strides=2)(inputs)
        conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(conv3)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(conv5)
        upsample1 = UpSampling2D((2, 2))(conv6)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsample1)
        upsample2 = UpSampling2D((2, 2))(conv7)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(upsample2)
        upsample3 = UpSampling2D((2, 2))(conv8)
        conv9=Conv2D(2, (3, 3), activation='tanh', padding='same')(upsample3)

        model = Model(inputs=inputs, outputs=conv9)
        # Finish model ===========================================================
        model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def GetNextBatch(self):
        if self.current_batch == self.train_batches:
                batch_files = self.train_files[self.batch_sz * self.current_batch:]
                images = self.DoReadImages(batch_files)
                X, Y = self.DoPreprocess(images)
                self.current_batch=0
                return X, Y
        self.current_batch_files = self.train_files[self.batch_sz * self.current_batch:self.batch_sz * (self.current_batch + 1)]
        #print(self.current_batch_files)
        images = self.DoReadImages(self.current_batch_files)
        X, Y = self.DoPreprocess(images)
        self.current_batch=self.current_batch+1
        return X,Y

    def DoTrain(self,model):
        self.epochs=5000
        for e in range(self.epochs):
            for b in range(self.train_batches):
                X,Y=self.GetNextBatch()
                #Train the neural network
                #model.fit(x=X, y=Y, batch_size=1, epochs=50000)
                model.train_on_batch(x=X, y=Y, sample_weight=None, class_weight=None)
                print("loss,accuracy",model.evaluate(X, Y, batch_size=1))
                print("[" + str(b + 1) + "/" + str(e + 1) + "/" + str(self.epochs) + "]")
        return model,X

    def DoTest(self,model,X,i):
        # Output colorizations
        output = model.predict(X)
        output = output * 128
        canvas = np.zeros((img_rows, img_cols, 3))
        canvas[:,:,0] = X[:,:,:,0]
        canvas[:,:,1:] = output[0]
        imsave("images/preds/comp_img_rgb_output_"+str(i)+".png", lab2rgb(canvas))
        imsave("images/preds/comp_img_gray_input_"+str(i)+".png", rgb2gray(lab2rgb(canvas)))

if __name__=='__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    bt=batching(train_dir="images/train/",batch_sz=4)
    test=1
    if test==1:
        model=bt.DoModel_Model()
        model,X=bt.DoTrain(model)
        for i in range(bt.batch_sz):
            a=X[i]
            #plt.imshow(a[:,:,0])
            #plt.show()
            img = a.reshape(1, bt.img_rows, bt.img_cols, 1)
            bt.DoTest(model,img,i)