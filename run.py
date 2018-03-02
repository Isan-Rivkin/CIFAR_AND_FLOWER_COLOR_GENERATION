import keras
import os
import numpy as np
import cv2
map = dict()
map[0] = "Bird"
map[1] = "Cat"
map[2] = "Dog"
map[3] = "Frog"
map[4] = "Horse"
map[10] = "Flower"


#params
DEFAULT_SRC = 'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\images'
#DEFAULT_SRC = r'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\testing\flowers'
#load model
#model_path = 'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\model\weights-05-0.50.h5'
model_path = r'D:\JetBrains\PyCharm 2017.1.1\AllProjects\CIFAR\testing\weights-05-0.70.h5'
model = keras.models.load_model(model_path)
#load images

pre_images  = []
names = []
for im_name in os.listdir(DEFAULT_SRC):
    names.append(im_name)
    img = cv2.imread(DEFAULT_SRC+'/'+im_name)
    img = cv2.resize(img,(32,32))
    pre_images.append(img)

images = np.asarray(pre_images)
predictions = model.predict_classes(images)
print(names)
print('-----------------------------------------------------------------')
for i in range(0,len(predictions)):
    print("{0} classified as => {1}".format(names[i],map[predictions[i]]))




