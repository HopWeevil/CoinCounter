import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ImageUtlis as iu
import CoinCounter
from sklearn.metrics import confusion_matrix
import glob
import re
import csv

CSVNAME = 'Coins.csv'
# Load data from CSV file
data = pd.read_csv(CSVNAME)

def Train(data, drawplot):
    X = np.array(data['Area']).astype(np.float32)
    y = np.array(data['CoinType']).astype(np.int32)
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)

    X_train, X_test = np.split(X, [train_size])
    y_train, y_test = np.split(y, [train_size])

    # Create SVM classifier
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_EPS, 100, 1e-6))

    # Train SVM classifier
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    svm.save('svm_model.dat')

    # Validate the model using the testing set
    y_pred = svm.predict(X_test)[1].ravel()
    accuracy = np.mean(y_pred == y_test) * 100
    print("SVM accuracy: {:.2f}%".format(accuracy))
    
    if drawplot:
        cm = confusion_matrix(y_test, y_pred)
        # Visualize confusion matrix
        fig, ax = plt.subplots()
        # Set background color to white
        ax.imshow(cm, cmap=plt.cm.Oranges)
        ax.set_xticks(np.arange(len(np.unique(y_test))))
        ax.set_yticks(np.arange(len(np.unique(y_test))))
        ax.set_xticklabels(np.unique(y_test))
        ax.set_yticklabels(np.unique(y_test))
        for i in range(len(np.unique(y_test))):
            for j in range(len(np.unique(y_test))):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.xlabel('Predicted Coin Type')
        plt.ylabel('True Coin Type')

        plt.show()

#Train(data,False)


def AddCoinRecords(records):
    coin_order = [10, 5, 25, 50, 1]
    
    sorted_records = sorted(records, key=lambda x: coin_order.index(int(x[0])) if int(x[0]) in coin_order else len(coin_order))

    with open('Coins.csv', 'a', newline='') as file:
        writer = csv.writer(file)
    
        for record in sorted_records:
            writer.writerow(record)