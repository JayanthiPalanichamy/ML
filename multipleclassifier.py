from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

##decision tree
# from sklearn import tree
# my_classifier=tree.DecisionTreeClassifier()
##nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
my_classifier=KNeighborsClassifier()

my_classifier.fit(x_train,y_train)
predictions=my_classifier.predict(x_test)

##testing the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
