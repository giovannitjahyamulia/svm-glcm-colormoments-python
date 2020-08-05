from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import numpy as np
import warnings

accuracy = 0

def svm(X_train, X_test, y_train, y_test, kernelSVM, feature):
    warnings.filterwarnings('always')
    
    svm_model = SVC(kernel = kernelSVM).fit(X_train, y_train) 
    
    y_predict = svm_model.predict(X_test)

    # creating a confusion matrix 
    # cm = confusion_matrix(y_test, y_predict)
    print("\nSVM Classification using", feature, "feature and", kernelSVM, "kernel.\n")
    
    filename = "SVM-" + feature + "_" + kernelSVM
    
    print("\nConfusion Matrix SVM Classification using", feature, "feature and", kernelSVM, "kernel.")
    confusionmatrix = confusion_matrix(y_test, y_predict)
    print(confusionmatrix)
    np.savetxt(filename + '_confusion_matrix.csv', confusionmatrix.astype(int), fmt='%i', delimiter="   ")

    print("\nClassification Report SVM Classification using", feature, "feature and", kernelSVM, "kernel.")
    classificationreport = classification_report(y_test, y_predict)
    print(classificationreport)
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", svm_model.score(X_test, y_test))

    # Model Precision: what percentage of positive tuples are labeled as such?
    # Model Recall: what percentage of positive tuples are labelled as such?


    report_file = open(filename + '_classification_report.csv', "w")
    report_file.write(
        classificationreport + "\n\n" +
        "Accuracy: " + str(svm_model.score(X_test, y_test))
    )