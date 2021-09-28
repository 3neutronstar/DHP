# -*- coding: utf-8 -*-
from pandas.core.frame import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from solution import Solution
from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost
class FsProblem :
    def __init__(self,typeOfAlgo,data,clinical_data,qlearn,classifier=KNeighborsClassifier(n_neighbors=1)):
        self.data=data
        self.nb_attribs = len(self.data.columns)-1 # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts=self.data.iloc[:,self.nb_attribs] # We initilize the labels from the last column of the dataset # 마지막 column이 정답
        self.ql = qlearn
        # self.classifier = classifier # classifier 대신에 xgboost같은거 넣으면 됨
        # self.classifier = LinearRegression()
        self.classifier = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=9)
        self.typeOfAlgo = typeOfAlgo
        self.clinical_variable_data=clinical_data.values

    def evaluate2(self,solution):
        sol_list = Solution.sol_to_list(solution)
        if (len(sol_list) == 0):
            return 0
         
        df = self.data.iloc[:,sol_list]
        array=df.values
        X = array[:,0:self.nb_attribs] # clinical variable 추가 위치
        X = np.concatenate((X,self.clinical_variable_data),axis=1)
        Y = np.log(self.outPuts)
        train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    random_state=0,
                                                    test_size=0.1
                                                    )
        self.classifier.fit(train_X,train_y)
        predict = self.classifier.predict(test_X) 
        return -metrics.mean_squared_error(Y,predict)
        # return metrics.accuracy_score(predict,test_y)


    def evaluate(self,solution):
        sol_list = Solution.sol_to_list(solution)
        if (len(sol_list) == 0):
            return 0
        
        df = self.data.iloc[:,sol_list] # For this function you need to put the indexes of features you picked  
        array=df.values
        X = array[:, 0:self.nb_attribs] # clinical variable 추가 위치
        X = np.concatenate((X,self.clinical_variable_data),axis=1)
        Y = np.log(self.outPuts)
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0) # Cross validation function
        # results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='accuracy')
        results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='neg_mean_squared_error')
        #print("\n[Cross validation results]\n{0}".format(results))
        return results.mean()
