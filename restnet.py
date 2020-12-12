import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import requests

from PIL import Image
from io import BytesIO

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.applications.resnet50 import ResNet50
from keras.callbacks import LearningRateScheduler
def plot_cm(label, preds, n_classes=20):

    cm = confusion_matrix(label, preds)
    def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
        plt.rc('font',family='Times New Roman',size='5')   
        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j]*100 + 0.5) == 0:
                    cm[i, j]=0

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Actual',
            xlabel='Predicted')

        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j]*100 + 0.5) > 0:
                    ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                            ha="center", va="center",
                            color="white"  if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig('ResNet.jpg', dpi=300)
        plt.show()

    plot_Matrix(cm, range(n_classes))

def scheduler(epoch):
    
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


warnings.filterwarnings('ignore')


path = './training_data' 
classes = os.listdir(path) 
print(f'Total number of categories: {len(classes)}')

counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))
    
print(f'Total number of images in dataset: {sum(list(counts.values()))}')

imbalanced = sorted(counts.items(), key = lambda x: x[1], reverse = True)[:20]
print(imbalanced)

imbalanced = [i[0] for i in imbalanced]
print(imbalanced)

X = [] 
Y = [] 

for c in classes:
  
    if c in imbalanced:
        dir_path = os.path.join(path, c)
        label = imbalanced.index(c) 
        
        for i in os.listdir(dir_path):
            image = cv.imread(os.path.join(dir_path, i))
            
            try:
                resized = cv.resize(image, (96, 96)) 
                X.append(resized)
                Y.append(label)
            
            except:
                print(os.path.join(dir_path, i), '[ERROR] can\'t read the file')
                continue       
            
print('DONE')


X = np.array(X).reshape(-1, 96, 96, 3)

X = X / 255.0

y = to_categorical(Y, num_classes = len(imbalanced))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, shuffle = True, random_state = 666)
reduce_lr = LearningRateScheduler(scheduler)


datagen = ImageDataGenerator(rotation_range = 45,
                            zoom_range = 0.2, 
                            horizontal_flip = True, 
                            width_shift_range = 0.15, 
                            height_shift_range = 0.15, 
                            shear_range = 0.2) 

datagen.fit(X_train)


model = ResNet50(weights=None, include_top=True, input_shape=(96, 96, 3), classes=20)

checkpoint = ModelCheckpoint('./CNN_model/best_model.hdf5', verbose = 1, monitor = 'accuracy', save_best_only = True)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 20), epochs = 100, validation_data = [X_test, y_test],
                             steps_per_epoch=len(X_train) // 20, callbacks = [checkpoint,reduce_lr])



import os
if not os.path.exists('./resnet_model'):
    os.mkdir('./resnet_model')
model.save('./resnet_model/model.hdf5')

model.load_weights('./resnet_model/model.hdf5')

pred_y = np.argmax(model.predict(X_test),1)
y_test = np.argmax(y_test,1)
fig = plt.figure()
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='train_loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('ResNet_loss.jpg', dpi=300)

fig = plt.figure()
epochs = len(history.history['acc'])
plt.plot(range(epochs), history.history['acc'], label='train_acc')
plt.plot(range(epochs), history.history['val_acc'], label='val_acc')
plt.legend()
plt.savefig('ResNet_acc.jpg', dpi=300)
plt.show()
plot_cm(y_test, pred_y)



