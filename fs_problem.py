# -*- coding: utf-8 -*-
from pandas.core.frame import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from solution import Solution
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import shap
import pickle
import xgboost
import joblib
import os
import matplotlib.pyplot as plt


class FsProblem:
    def __init__(self, typeOfAlgo, data, clinical_data, qlearn, classifier, log_dir):
        self.data = data
        self.nb_attribs = len(
            self.data.columns) - 1  # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts = self.data.iloc[:,
                       self.nb_attribs]  # We initilize the labels from the last column of the dataset # 마지막 column이 정답
        self.ql = qlearn
        self.cv_n_split = 10
        # self.classifier = classifier # classifier 대신에 xgboost같은거 넣으면 됨
        self.classifier_name = classifier
        self.classifier = self._set_model(self.classifier_name) # LinearRegression()
        self.typeOfAlgo = typeOfAlgo
        self.clinical_variable_data = clinical_data.values
        self.test_size = 0.1
        self.log_dir = log_dir

    def _prepare_data(self, solution, cross_validation_flag=False, clinic_var=False):
        '''
        현재 state의 solution 상태를 보고 적절한 데이터 구조로 x, y 데이터와 구체적인 split 정보를 반환해준다.
        cross_validation_flag가 True면 split_info는 cross validation policy 반환.
        cross_validation_flag가 False면 split_info는 (train_x, test_x, train_y, test_y) 반환.

        :param solution: 현재 Agent의 solution.
        :param cross_validation_flag: cross validation이 적용된 정보를 반환할지 선택.
        :return: 전체 x, 전체 y, cross validation 조건에 맞는 정보
        '''
        sol_list = Solution.sol_to_list(solution)

        if len(sol_list) == 0:
            return None, None, None

        # For this function you need to put the indexes of features you picked
        df = self.data.iloc[:, sol_list]
        array = df.values

        gene_x = array[:, 0:self.nb_attribs] # clinical variable 추가 위치
        if clinic_var:
            total_x = np.concatenate((gene_x, self.clinical_variable_data), axis=1)
        else:
            total_x = gene_x

        total_y = np.log(self.outPuts)

        if cross_validation_flag:
            split_info = ShuffleSplit(n_splits=self.cv_n_split, test_size=self.test_size, random_state=0)
        else:
            split_info = train_test_split(total_x, total_y, random_state=0, test_size=self.test_size)

        return total_x, total_y, split_info

    def evaluate(self, solution, train=True):
        total_x, total_y, split_info = self._prepare_data(solution, cross_validation_flag=False, clinic_var=train)

        self.classifier = self._set_model(type=self.classifier_name)

        if total_x is None:
            return 0

        train_x, test_x, train_y, test_y = split_info

        self.classifier.fit(train_x, train_y)
        predict = self.classifier.predict(test_x)

        # metrics.accuracy_score(predict,test_y)
        reward = 1.0 / metrics.mean_squared_error(test_y, predict)

        if not train:
            self.get_shap_value(train_x, test_x)
            self._save_model()

        return reward

        # results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='accuracy')

    def get_shap_value(self, train_x, test_x):
        '''
        Agent의 최종 solution을 이용하여 모델을 학습하고 validation 데이터 셋에 shap value를 시각화하여 보여줌.
        :param test_x: 테스트용 데이터
        :param train_x: 학습용 데이터
        :return: 없음.
        '''
        # 그래프 초기화
        shap.initjs()

        ex = shap.KernelExplainer(self.classifier.predict, train_x)

        # 첫번째 test dataset 하나에 대해서 shap value를 적용하여 시각화
        shap_values = ex.shap_values(test_x[0, :])
        shap.force_plot(ex.expected_value, shap_values, test_x[0, :])

        # 전체 검증 데이터 셋에 대해서 적용
        shap_values = ex.shap_values(test_x)
        shap.summary_plot(shap_values, test_x, show=False)

        plt.savefig(os.path.join(self.log_dir, 'shap.png'))

    def _set_model(self, type):
        if type == 'logistic_regression':
            model = LogisticRegression()
            params = {
                'penalty': ['none', 'l1', 'elasticnet'],
                'solver': ['lbfgs', 'saga'],
                'l1_ratio': [0.5],
                'max_iter': [1000]
            }
            return GridSearchCV(model, param_grid=params, cv=self.cv_n_split)

        elif type == 'linear_regression':
            return LinearRegression()
        elif type == 'lightgbm':
            model = LGBMRegressor()

            params = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [-1, 3, 5, 7, 15],
                'num_leaves': [7, 14, 21, 28, 31, 50],
                'learning_rate': [0.1, 0.03, 0.003],
            }
            return GridSearchCV(model, param_grid=params, cv=self.cv_n_split)

    def _save_model(self):
        joblib.dump(self.classifier, os.path.join(self.log_dir, 'model.pkl'))

