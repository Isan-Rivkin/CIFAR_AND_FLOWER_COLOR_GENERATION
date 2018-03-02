import numpy
import cv2
import os

# resize images to (x,y)
def resize_images(src,dest,resize):
    resized_images = []
    names = []
    for f in os.listdir(src):
        img = cv2.imread(src + '/' + f)
        img = cv2.resize(img, resize)
        cv2.imwrite(dest + '/' + f , img)

HQ_SRC = r'D:\MLIternshipMihlala\flowers\raw_images\JPEG96H'
SMALL_DEST = r'D:\MLIternshipMihlala\flowers\raw_images\JPEG32'
LQ_DEST = r'D:\MLIternshipMihlala\flowers\raw_images\JPEG96L'
resize_images(HQ_SRC,SMALL_DEST,(32,32))
resize_images(SMALL_DEST,LQ_DEST,(96,96))