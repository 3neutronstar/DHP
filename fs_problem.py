# -*- coding: utf-8 -*-
from pandas.core.frame import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor
from solution import Solution
from sklearn.linear_model import LinearRegression
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt

class FsProblem:
    def __init__(self, typeOfAlgo, data, clinical_data, qlearn, classifier=KNeighborsClassifier(n_neighbors=1)):
        self.data = data
        self.nb_attribs = len(
            self.data.columns) - 1  # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts = self.data.iloc[:,
                       self.nb_attribs]  # We initilize the labels from the last column of the dataset # 마지막 column이 정답
        self.ql = qlearn
        # self.classifier = classifier # classifier 대신에 xgboost같은거 넣으면 됨
        if classifier=='linear':
            self.classifier = LinearRegression()
        elif classifier=='xgb':
            self.classifier = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=9)
        self.typeOfAlgo = typeOfAlgo
        self.clinical_variable_data = clinical_data.values
        self.test_size = 0.1
        self.cv_n_split = 10

    def _prepare_data(self, solution, cross_validation_flag=False, train=False):
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

        gene_x = array[:, 0:self.nb_attribs]  # clinical variable 추가 위치
        if train:
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
        total_x, total_y, split_info = self._prepare_data(solution, cross_validation_flag=train, train=train)

        if total_x is None:
            return 0

        if train:
            cv = split_info
            results = -1.0 / cross_val_score(self.classifier, total_x, total_y,
                                             cv=cv, scoring='neg_mean_squared_error')
            reward = results.mean()
        else:
            train_x, test_x, train_y, test_y = split_info

            self.classifier.fit(train_x, train_y)
            predict = self.classifier.predict(test_x)

            # metrics.accuracy_score(predict,test_y)
            reward = 1.0 / metrics.mean_squared_error(test_y, predict)

            self.get_shap_value(train_x, test_x)

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
        if isinstance(self.classifier,LinearRegression):
            shap.initjs()

            ex = shap.KernelExplainer(self.classifier.predict, train_x)

            # 첫번째 test dataset 하나에 대해서 shap value를 적용하여 시각화
            shap_values = ex.shap_values(test_x[0, :])
            shap.force_plot(ex.expected_value, shap_values, test_x[0, :])

            # 전체 검증 데이터 셋에 대해서 적용
            shap_values = ex.shap_values(test_x)
            shap.summary_plot(shap_values, test_x)
            plt.savefig('./linear_regression_result.jpg')
        elif isinstance(self.classifier,XGBRegressor):
            self.classifier.plot_importance()
            plt.savefig('./xgboost_importance_result.jpg')
            plt.close()

            explainer = shap.Explainer(self.classifier)
            shap_values = explainer(test_x[0,:])

            # visualize the first prediction's explanation
            shap.plots.waterfall(shap_values[0])
            plt.savefig('./xgboost_waterfall_result.jpg')

