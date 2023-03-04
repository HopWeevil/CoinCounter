
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

CSVNAME = 'Coins.csv'
IMGSPATH = 'Images/'
data = pd.read_csv(CSVNAME)

def Train(data, drawplot):
    # Extract features and labels from data
    X = np.array(data[['Area', 'b', 'g', 'r']]).astype(np.float32)
    y = np.array(data['CoinType']).astype(np.int32)
    img_names = np.array(data['imgname']).astype(str) # Add image names

    # Split the data into training and testing sets
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)

    X_train, X_test = np.split(X, [train_size])
    y_train, y_test = np.split(y, [train_size])
    img_names_test = img_names[train_size:] # Extract image names of testing set

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

    incorrect_pred = y_pred != y_test # Boolean array indicating incorrect predictions
    incorrect_names = img_names_test[incorrect_pred] # Extract names of incorrectly predicted images

    if len(incorrect_names) > 0:
        print("Incorrectly predicted images:")
        print(incorrect_names)

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

    return svm

def PrepairData(svm):
    csv = pd.read_csv(CSVNAME)
    files = glob.glob1(IMGSPATH,'*'+".jpg")
    files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    
    for filename in files:
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(IMGSPATH, filename))
            warpedimage = iu.WarpImage(img, 3, False)
            imgPre = iu.ImagePre(warpedimage, False)
            records = CoinCounter.CountCoins(imgPre, warpedimage,svm,1,2,False)[1]
            coin_order = [10, 5, 25, 50, 1]
            records = sorted(records, key=lambda x: coin_order.index(x[0]))
            for record in records:
                record.append(filename)
                if filename not in csv["imgname"].values:
                    data = np.array(record)
                    data = data.reshape(1, -1)
                    df = pd.DataFrame(data, columns=["CoinType", "Area", "b", "g", "r", "imgname"])
                    df.to_csv(CSVNAME, mode='a', index=False, header=False)

svm = Train(data, drawplot=True)
#PrepairData(svm)