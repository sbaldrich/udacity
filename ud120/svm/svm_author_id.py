#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#clf = SVC(kernel = "linear") Train using 1% of the data and an rbf kernel

# Toss out 99% of the data
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

# Find the optimal value for C
#for c in [10, 100, 1000, 10000]:
for c in [10000]: # only use C = 10000 b/c it was found to be its optimal value
    clf = SVC(kernel = "rbf", C = c)
    t0 = time()
    clf.fit(features_train, labels_train)
    print "Training time for SVM with C = %.1f is: %.3f" % (c, round(time() - t0, 3))
    t0 = time()
    pred = clf.predict(features_test)
    print "Prediction time for SVM is with C = %.1f is: %.3f" % (c, round(time() - t0, 3))
    print "The score of the SVM model with C = %.1f is: %.3f" % (c, accuracy_score(pred, labels_test))
print "Predictions for elements(10, 26, 50) = (%d, %d, %d)" % (pred[10], pred[26], pred[50])
print "Number of events predicted as being on Chris(1) class: %d" % (len([x for x in pred if x]))
#########################################################


