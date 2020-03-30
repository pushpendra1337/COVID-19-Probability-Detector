'''
File: training.py
Description: Code for training the model using logistic regression

'''
__author__ = "Pushpendra Yadav"
__credits__ = ["Pushpendra Yadav"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Pushpendra Yadav"
__email__ = "pushpendray1337@gmail.com"


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":

	#Data read
	df = pd.read_csv('data.csv')
	train, test = data_split(df, 0.2)
	X_train = train[['age', 'bodyTemp', 'dryCough', 'sneezing', 'soreThroat', 'weakness', 'severeCough', 'diffBreath', 'pain', 'travelHist', 'contactWith', 'runnyNose', 'diabetes', 'highBP', 'heartD', 'kidneyD', 'lungD', 'lessImmune']].to_numpy()
	X_test = test[['age', 'bodyTemp', 'dryCough', 'sneezing', 'soreThroat', 'weakness', 'severeCough', 'diffBreath', 'pain', 'travelHist', 'contactWith', 'runnyNose', 'diabetes', 'highBP', 'heartD', 'kidneyD', 'lungD', 'lessImmune']].to_numpy()

	Y_train = train[['infProb']].to_numpy().reshape(148,)
	Y_test = test[['infProb']].to_numpy().reshape(36,)

	clf = LogisticRegression()
	clf.fit(X_train, Y_train)

	#Open a file, where you want to store the data
	file = open('model.pkl', 'wb')

	#Dump info to that file
	pickle.dump(clf, file)
	file.close()