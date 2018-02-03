# from sklearn import datasets
# digits=datasets.load_digits()
# x=digits.data
# y=digits.target
# print(x[0])
from sklearn.datasets import make_regression
# Generate features matrix, target vector, and the true coefficients
X, y, coef = make_regression(n_samples = 100,n_features = 3,n_informative = 3,n_targets = 1,noise = 0.0,coef = True,random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', X[:3])
print('Target Vector\n', y[:3])
