from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import keras
import cv2
from PIL import Image
SRC = r'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\Flowers\bw_flowers'
DEST = 'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\Flowers\painted_flowers'
MODEL_PATH = 'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\Flowers\FlowerModel\FlowersFinalModel.h5'
DEST_RESIZED = 'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\Flowers\scaled_up'
model = keras.models.load_model(MODEL_PATH)


def resize_images(SRC,DEST,resize):
    resize_to = resize
    for f in os.listdir(SRC):
        image = Image.open(SRC+ '/' + f)
        new_im = image.resize(resize_to, Image.ANTIALIAS)
        new_im.save(DEST + '/' + f)

def predict_images(model,SRC,DEST):
    color_me = []
    for filename in os.listdir(SRC):
        m = load_img(SRC + '/' + filename)
        color_me.append(img_to_array(m))

    color_me = np.array(color_me, dtype = float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))

    # Test model
    output =model.predict(color_me)
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((32, 32, 3))
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        imsave(DEST + '/'+str(i) + ".png", lab2rgb(cur))



resize_images(SRC,SRC,(32,32))
predict_images(model,SRC,DEST)
resize_images(DEST,DEST_RESIZED,(96,96))
