# from sklearn.datasets import load_iris
# iris=load_iris()
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0:2])
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
iris=load_iris()
test_index=[0,50,100]

 #training dataset
train_target=np.delete(iris.target,test_index)
train_data=np.delete(iris.data,test_index,axis=0)


#testing dataset
test_target=iris.target[test_index]
test_data=iris.data[test_index]
clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#checking the test dataset
##print(test_target==clf.predict(test_data))

#visualize tree
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("irisColoured.pdf")
print(test_data[1],test_target[1])
print(iris.feature_names,iris.target_names)
