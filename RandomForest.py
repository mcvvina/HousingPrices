from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import sys

def Regressor(x_data, y_data, test = []):
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3)


    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_features = "sqrt", warm_start=True)

    rf.fit(X_train, y_train)
    printStatements = []
    
    predictions = rf.predict(X_test)
   
    rf.n_estimators = 2000
    rf.fit(x_data, y_data)
    rf.n_estimators = 3000
    rf.fit(x_data, y_data)
    
    printStatements.append(rf.score(X_train,y_train))
    printStatements.append(rf.score(X_test,y_test))

    if test ==[]:
        pass

    return printStatements #rf.score(train), rf.score(test)
    


def Classifier(x_data, y_data):
    pass
