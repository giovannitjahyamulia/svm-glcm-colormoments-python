from PIL import Image
import numpy as np
import skimage
import skimage.io
import skimage.feature
from skimage.feature import greycoprops
import math
import statistics
import scipy
import scipy.stats
from scipy.stats import skew
from decimal import Decimal

from numpy import genfromtxt

import os
import svm_multiclass

from scipy import misc

train_feature = []
train_label = []
test_feature = []
test_label = []

root_train = 'C:/Users/Acer/Documents/Gio/Skripsi/image_train'
root_test = 'C:/Users/Acer/Documents/Gio/Skripsi/image_test'

def train(parent):
    print("Training started!")

    folder_index = 0
    image_index = 0
    image_total = 0

    feature = []
    label = []
    index = 0

    for folder in os.listdir(parent):
        index = index + 1
        current_path = "".join((parent, "/", folder))

        print("Extracting Feature from", folder)

        for file in os.listdir(current_path):
            path = (current_path + "/" + file)

            img = skimage.io.imread(path, as_gray=True)
            img = skimage.img_as_ubyte(img)
            img = np.asarray(img, dtype="int32")
            
            g = skimage.feature.greycomatrix(img, [1], [np.pi/2], levels=img.max()+1, symmetric=False, normed=True)
            glcm_contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
            glcm_energy = skimage.feature.greycoprops(g, 'energy')[0][0]
            glcm_homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
            glcm_correlation = skimage.feature.greycoprops(g, 'correlation')[0][0]

            img = skimage.io.imread(path, as_gray=False)
            img = np.asarray(img)

            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]
                       
            meanR = np.mean(R)
            meanG = np.mean(G)
            meanB = np.mean(B)

            varianceR = np.var(R)
            varianceG = np.var(G)
            varianceB = np.var(B)
            
            differenceR = 0.0
            differenceG = 0.0
            differenceB = 0.0

            for i in range (len(img)):
                for j in range (len(img[0])):
                    differenceR = differenceR - np.float_power((R[i][j] - meanR), 3)
                    differenceG = differenceG - np.float_power((G[i][j] - meanG), 3)
                    differenceB = differenceB - np.float_power((B[i][j] - meanB), 3)

            N = len(img) * len(img[0])

            skewnessR = np.float_power((differenceR/N), 1/3.)
            skewnessG = np.float_power((differenceG/N), 1/3.)
            skewnessB = np.float_power((differenceB/N), 1/3.)

            if not glcm_contrast is None or not glcm_energy is None or not glcm_homogeneity is None or not glcm_correlation is None or not meanR is None or not meanG is None or not meanB is None or not varianceR is None or not varianceG is None or not varianceB is None or not skewnessR is None or not skewnessG is None or not skewnessB is None:
                temp = [glcm_contrast, glcm_energy, glcm_homogeneity, glcm_correlation, meanR, meanG, meanB, varianceR, varianceG, varianceB, skewnessR, skewnessG, skewnessB]
                # print(temp)
                train_feature.append(temp)
                train_label.append(index)
        np.savetxt("SVM-GLCM-90-Color Moments_train_feature.csv", train_feature, delimiter=",")
        np.savetxt("SVM-GLCM-90-Color Moments_train_label.csv", train_label, delimiter=",")
    print("Training finish...")
    
def test(parent):
    print("Testing started!")
    folder_index = 0
    image_index = 0
    image_total = 0

    feature = []
    label = []
    index = 0

    for folder in os.listdir(parent):
        index = index + 1
        current_path = "".join((parent, "/", folder))

        print("Extracting Feature from", folder)

        for file in os.listdir(current_path):
            path = (current_path + "/" + file)

            img = skimage.io.imread(path, as_gray=True)
            img = skimage.img_as_ubyte(img)
            img = np.asarray(img, dtype="int32")

            g = skimage.feature.greycomatrix(img, [1], [np.pi/2], levels=img.max()+1, symmetric=False, normed=True)
            glcm_contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
            glcm_energy = skimage.feature.greycoprops(g, 'energy')[0][0]
            glcm_homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
            glcm_correlation = skimage.feature.greycoprops(g, 'correlation')[0][0]

            img = skimage.io.imread(path, as_gray=False)
            img = np.asarray(img)

            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]
                       
            meanR = np.mean(R)
            meanG = np.mean(G)
            meanB = np.mean(B)

            varianceR = np.var(R)
            varianceG = np.var(G)
            varianceB = np.var(B)
            
            differenceR = 0.0
            differenceG = 0.0
            differenceB = 0.0

            for i in range (len(img)):
                for j in range (len(img[0])):
                    differenceR = differenceR - np.float_power((R[i][j] - meanR), 3)
                    differenceG = differenceG - np.float_power((G[i][j] - meanG), 3)
                    differenceB = differenceB - np.float_power((B[i][j] - meanB), 3)

            N = len(img) * len(img[0])

            skewnessR = np.float_power((differenceR/N), 1/3.)
            skewnessG = np.float_power((differenceG/N), 1/3.)
            skewnessB = np.float_power((differenceB/N), 1/3.)

            if not glcm_contrast is None or not glcm_energy is None or not glcm_homogeneity is None or not glcm_correlation is None or not meanR is None or not meanG is None or not meanB is None or not varianceR is None or not varianceG is None or not varianceB is None or not skewnessR is None or not skewnessG is None or not skewnessB is None:
                temp = [glcm_contrast, glcm_energy, glcm_homogeneity, glcm_correlation, meanR, meanG, meanB, varianceR, varianceG, varianceB, skewnessR, skewnessG, skewnessB]
                # print(temp)
                test_feature.append(temp)
                test_label.append(index)
        np.savetxt("SVM-GLCM-90-Color Moments_test_feature.csv", test_feature, delimiter=",")
        np.savetxt("SVM-GLCM-90-Color Moments_test_label.csv", test_label, delimiter=",")
    print("Testing finish...")

def main():
    # X_train = genfromtxt('C:\\Gio\\PC-Riset\\Python\\SVM-GLCM-Color Moments_train_feature.csv', delimiter=',')
    # y_train = genfromtxt('C:\\Gio\\PC-Riset\Python\\SVM-GLCM-Color Moments_train_label.csv', delimiter=',')
    # X_train = np.nan_to_num(np.array(X_train))
    # y_train = np.nan_to_num(np.array(y_train)).astype(int)

    # X_test = genfromtxt('C:\\Gio\\PC-Riset\\Python\\SVM-GLCM-Color Moments_test_feature.csv', delimiter=',')
    # y_test = genfromtxt('C:\\Gio\\PC-Riset\Python\\SVM-GLCM-Color Moments_test_label.csv', delimiter=',')
    # X_test = np.nan_to_num(np.array(X_test))
    # y_test = np.nan_to_num(np.array(y_test)).astype(int)

    train(root_train)
    test(root_test)
    
    # X_train = np.nan_to_num(np.array(train_feature))
    # y_train = np.nan_to_num(np.array(train_label))
    # X_test = np.nan_to_num(np.array(test_feature))
    # y_test = np.nan_to_num(np.array(test_label))

    # print("\nTraining Features\n")
    # print("Training features with dimension:", X_train.shape)
    # print("Training label with dimension:", y_train.shape)

    # print("\nTest Features\n")
    # print("Test features with dimension:", X_test.shape)
    # print("Test label with dimension:", y_test.shape)

    # svm_multiclass.svm(X_train, X_test, y_train, y_test, 'linear', "GLCM-Color Moments")
    # svm_multiclass.svm(X_train, X_test, y_train, y_test, 'poly', "GLCM-Color Moments")
    # svm_multiclass.svm(X_train, X_test, y_train, y_test, 'sigmoid', "GLCM-Color Moments")
    # svm_multiclass.svm(X_train, X_test, y_train, y_test, 'rbf', "GLCM-Color Moments")
    

main()