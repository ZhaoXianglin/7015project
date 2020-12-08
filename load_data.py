import  os
from os import path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# 从文件夹加载数据
def load_data(path,width,layers=3):
    X = []
    y = []
    #载入数据文件夹,文件夹名是类名
    classes = os.listdir(path)
    # 计算类别,按照类别中图片数量自大到小排序
    counts = {}
    for c in classes:
        if c == '.DS_Store':
                continue
        counts[c] = len(os.listdir(os.path.join(path, c)))
    imbalanced = sorted(counts.items(), key = lambda x: x[1], reverse = True)
    print(imbalanced)

    for num,c in enumerate(imbalanced):
        # 忽略mac下的隐藏文件夹
        if c[0] == '.DS_Store':
                continue
        print("Class{0}:{1}".format(num+1,c[0]))
        #文件路径列表
        dir = os.path.join(path, c[0])
        #逐层读取文件
        childs = os.listdir(dir)
        for child in childs:
            if child == '.DS_Store':
                continue
            img_dir = os.path.join(dir,child)
            img = cv.imread(img_dir)
            X.append(img)
            y.append(num)
    # 转换为numpy数组
    X = np.array(X)
    print(X.shape)
    #标准化
    X = X / 225.0
    y = to_categorical(y, num_classes = 20)
    return X,y



if __name__ == '__main__':
    path = './Standard/RGB128'
    X,y = load_data(path,128,layers=3)
    print(X[0],y)
    
