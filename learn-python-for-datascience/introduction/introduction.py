from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Decision Tree
clf_tree = tree.DecisionTreeClassifier()

# Support Vector Machine
clf_svm = SVC()

# KNN Classifier
clf_knn = KNeighborsClassifier(n_neighbors=3)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Training
clf_tree = clf_tree.fit( X,Y )
clf_svm  = clf_svm.fit( X,Y )
clf_knn  = clf_knn.fit( X,Y )

prediction_tree = clf_tree.predict([[ 190, 70, 43]])
prediction_svm  = clf_svm.predict([[ 190, 70, 43]])
prediction_knn  = clf_knn.predict([[ 190, 70, 43]])

print ("Decision Tree - result %s" % prediction_tree)
print ("Support Vector Machine - result %s" % prediction_svm)
print ("K Nearest Neighbours - result %s" % prediction_knn)