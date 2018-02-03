from sklearn import tree
features=[[130,1],[140,1],[170,0],[150,0]]
labels=[0,0,1,1]
dict1={0:'Apple',1:'Orange'}
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
a=clf.predict([[160,0]])
print(dict1[a[0]])
