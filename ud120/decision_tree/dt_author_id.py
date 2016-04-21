#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split = 40)
t0 = time()
clf.fit(features_train, labels_train)
print "Training time for Decision Tree Classifier is: %.3f" % (time() - t0)
t0 = time()
pred = clf.predict(features_test)
print "Prediction time for Decision Tree Classifier is: %.3f" % (time() - t0)

from sklearn.metrics import accuracy_score
print "Accuracy score for Decision Tree Classifier is: %.3f" % accuracy_score(pred, labels_test)

#########################################################


