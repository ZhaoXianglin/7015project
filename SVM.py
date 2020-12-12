import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt

import warnings
import gc
import requests

from io import BytesIO

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve


warnings.filterwarnings('ignore')

def plot_cm(label, preds, n_classes=20):

    cm = confusion_matrix(label, preds)
    def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
        plt.rc('font',family='Times New Roman',size='8')   
        
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
        plt.savefig('SVM.jpg', dpi=300)
        plt.show()

    plot_Matrix(cm, range(n_classes))

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('learning curve.jpg', dpi=300)
    plt.show()


path = './training_data' 
classes = os.listdir(path) 
print(f'Total number of categories: {len(classes)}')

counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))
    
print(f'Total number of images in dataset: {sum(list(counts.values()))}')

imbalanced = sorted(counts.items(), key = lambda x: x[1], reverse = True)
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


X = np.array(X).reshape(-1, 96*96*3)

X = X / 255.0

y = to_categorical(Y, num_classes = len(imbalanced))
y = np.argmax(y,1)

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)

clf = SVC(kernel='linear',C=1,tol=0.0001,max_iter=100)
clf.fit(train_x,train_y)
import os
if not os.path.exists('./SVM_model'):
    os.mkdir('./SVM_model')

with open('./SVM_model/clf.model', 'wb') as f: 
    pickle.dump(clf, f)

with open('./SVM_model/clf.model', 'rb') as f: 
    clf = pickle.load(f)

    
pred_y = clf.predict(test_x)
print(classification_report(test_y,pred_y))
plot_cm(test_y, pred_y)

plot_learning_curve(clf, 'clf learning curve ', train_x, train_y)
