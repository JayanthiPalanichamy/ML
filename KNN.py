import random
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)
class ScarppyKNN():
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    def closest(self,item):
        best_distance=euc(item,self.x_train[0])
        best_index=0
        for i in range(1,len(x_train)):
            dist=euc(item,self.x_train[i])
            if dist < best_distance:
                best_distance=dist
                best_index=i
        return self.y_train[best_index]
    def predict(self,x_test):
        predictions=[]
        for item in x_test:
            label=self.closest(item)
            predictions.append(label)
        return predictions



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)


my_classifier=ScarppyKNN()

my_classifier.fit(x_train,y_train)
predictions=my_classifier.predict(x_test)

##testing the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
