import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics

# Pick the model
iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.LinearClassifier(n_classes=3, feature_columns=feature_columns)


# Train the model
classifier.fit( iris.data, iris.target, steps=200, batch_size=32 )

iris_predicitons = list( classifier.predict(iris.data,as_iterable=True) )

# Test the model
score = metrics.accuracy_score( iris.target, iris_predicitons )

print( "Accuracy: %f" % score )