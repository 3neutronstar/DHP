
# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from solution import Solution
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

class FsProblem:
    def __init__(self, typeOfAlgo, gene_data, qlearn,
                 classifier, reward_df, reward_clinic=None,config=None):
        self.gene_data = gene_data
        self.nb_attribs = len(
            self.gene_data.columns) - 1  # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts = self.gene_data.loc[:,
                       'time']  # We initilize the labels from the last column of the dataset # 마지막 column이 정답
        self.ql = qlearn
        # self.classifier = classifier # classifier 대신에 xgboost같은거 넣으면 됨
        self.classifier_name = classifier

        if self.classifier_name=='linear':
            self.classifier = LinearRegression()
        elif self.classifier_name == 'deep':
            self.classifier = None
        elif self.classifier_name == 'cox':
            self.classifier = None
        else:
            raise NotImplementedError
        self.typeOfAlgo = typeOfAlgo
        self.reward_clinic=reward_clinic
        self.test_size = 0.1
        self.cv_n_split = 5
        self.reward_df = reward_df
        self.config=config

    def evaluate(self, solution, train=True, feature_name=None):
        if train:
            if self.classifier_name == 'cox':
                reward = self.calcualte_reward(solution, train=train)

                if reward is None:
                    return

        else:
            if self.classifier_name == 'cox':
                reward = self.calcualte_reward(solution, train=train)

        return reward

        # results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='accuracy')
    def calcualte_reward(self, solution, train=True):
        
        sol_list = Solution.sol_to_list(solution)
        this_df=pd.concat([self.reward_df.iloc[:, sol_list], self.reward_df['Treatment'],
                               self.reward_df['time'], self.reward_df['event']], axis=1).dropna(axis=0)
        smallgene = pd.DataFrame.copy(this_df)
        t = smallgene["Treatment"] == 1
        f = smallgene["Treatment"] == 0

        smallgene = smallgene.drop("Treatment", axis=1)

        cox1 = CoxPHFitter() # 치료 받은 환자 데이터
        cox1.fit(smallgene[t], duration_col='time', event_col='event', show_progress=False)

        cox2 = CoxPHFitter() # 치료 안받은 환자 데이터
        cox2.fit(smallgene[f], duration_col='time', event_col='event', show_progress=False)

        if train:
            diff = cox2.params_ - cox1.params_
            diff.sort_values(ascending=False)

            return sum(diff[:10])
        else:
            # cross-validation
            cox_cv_result = k_fold_cross_validation(cox1, smallgene[t], duration_col='time', event_col='event', k=5,
                                                    scoring_method="concordance_index")
            print('C-index(cross-validation) = ', np.mean(cox_cv_result))
            cox_cv_result = k_fold_cross_validation(cox2, smallgene[f], duration_col='time', event_col='event', k=5,
                                                    scoring_method="concordance_index")
            print('C-index(cross-validation) = ', np.mean(cox_cv_result))

            self.cox_info(sol_list)

    def cox_info(self,sol_list):
        this_df=pd.concat([self.reward_df.iloc[:, sol_list],self.reward_clinic, self.reward_df['Treatment'],
                        self.reward_df['time'], self.reward_df['event']], axis=1).dropna(axis=0)
        smallgene = pd.DataFrame.copy(this_df)

        t = smallgene["Treatment"] == 1
        f = smallgene["Treatment"] == 0

        smallgene = smallgene.drop("Treatment", axis=1)

        cox1 = CoxPHFitter() # 치료 받은 환자 데이터
        cox1.fit(smallgene[t], duration_col='time', event_col='event', show_progress=False)

        cox2 = CoxPHFitter() # 치료 안받은 환자 데이터
        cox2.fit(smallgene[f], duration_col='time', event_col='event', show_progress=False)
        diff = cox2.params_ - cox1.params_
        print('optimal solution list',sol_list)
        sorted_indices=list(diff.sort_values(ascending=False).index)
        print(f'{len(sorted_indices)} Used Features :',sorted_indices)
        print('Top10 features',sorted_indices[:10])
