#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import accuracy_score


candidates = [("KNN", KNeighborsClassifier()), ("AdaBoost", AdaBoostClassifier()), ("SVM", SVC()),
("Random Forest", RandomForestClassifier()), ("Naive Bayes", GaussianNB()), ("Decision Trees", DecisionTreeClassifier())]

for algorithm_name, clf in candidates:
    training_time = time()
    clf.fit(features_train, labels_train)
    training_time = time() - training_time
    prediction_time = time()
    predictions = clf.predict(features_test)
    prediction_time = time() - prediction_time
    accuracy = accuracy_score(predictions, labels_test)
    print "Times for %s classifier: [ Train: %.3f, Predict: %.3f ], accuracy: %.4f" % (algorithm_name, training_time, prediction_time, accuracy)

# From the results shown from the benchmark above, it seems that AdaBoost and Random Forests give the best performance for these data. Let's play a little with them:

for splits in [20, 30, 40, 50, 100]:
    clf = RandomForestClassifier(min_samples_split = splits)
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    accuracy = accuracy_score(predictions, labels_test)
    print "Accuracy for Random Forest Classifier with min_samples_split = %d is %.3f" %(splits, accuracy) 

#try:
#    prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass
