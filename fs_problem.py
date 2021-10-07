# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from solution import Solution
from sklearn.linear_model import LinearRegression
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Model
from torch.utils.data import DataLoader, TensorDataset
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

from utils import AverageMeter, ProgressMeter

class FsProblem:
    def __init__(self, typeOfAlgo, data, clinical_data, qlearn,
                 classifier=KNeighborsClassifier(n_neighbors=1), reward_df=None,config=None):
        self.data = data
        self.nb_attribs = len(
            self.data.columns) - 1  # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts = self.data.loc[:,
                       'time']  # We initilize the labels from the last column of the dataset # 마지막 column이 정답
        self.ql = qlearn
        print(data)
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
        self.clinical_variable_data = clinical_data.values
        self.test_size = 0.1
        self.cv_n_split = 5
        self.reward_df = reward_df
        self.config=config

    def _prepare_data(self, solution, cross_validation_flag=False, clinic_include=False):
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
        if clinic_include:
            total_x = np.concatenate((gene_x, self.clinical_variable_data), axis=1)
        else:
            total_x = gene_x.to_numpy()

        total_y = np.log(self.outPuts).to_numpy()

        if cross_validation_flag:
            split_info = ShuffleSplit(n_splits=self.cv_n_split, test_size=self.test_size, random_state=0)
        else:
            split_info = train_test_split(total_x, total_y, random_state=0, test_size=self.test_size)

        total_y = total_y.reshape(total_y.shape[0], 1)

        return total_x, total_y, split_info

    def evaluate(self, solution, train=True, feature_name=None):
        if train:
            if self.classifier_name == 'linear':
                total_x, total_y, split_info = self._prepare_data(solution, cross_validation_flag=True, clinic_include=train)
                if total_x is None:
                    return 0

                cv = split_info
                results = -1.0 / cross_val_score(self.classifier, total_x, total_y,
                                                 cv=cv, scoring='neg_mean_squared_error')
                reward = results.mean()
            elif self.classifier_name == 'deep':

                total_x, total_y, split_info = self._prepare_data(solution, cross_validation_flag=False, clinic_include=True)

                if total_x is None:
                    return 0

                loss = self.train_model(solution, total_x, total_y)

                reward = 1.0 / (loss+1e-8)
            elif self.classifier_name == 'cox':
                reward = self.calcualte_reward(solution, train=train)

                if reward is None:
                    return

        else:
            total_x, total_y, split_info = self._prepare_data(solution, cross_validation_flag=False, clinic_include=True)

            train_x, test_x, train_y, test_y = split_info

            if self.classifier_name == 'linear':
                self.classifier.fit(train_x, train_y)
                predict = self.classifier.predict(test_x)

                # metrics.accuracy_score(predict,test_y)
                reward = 0  # = 1.0 / metrics.mean_squared_error(test_y, predict)

                self.get_shap_value(train_x, test_x)
            elif self.classifier_name == 'deep':
                self.train_best_solution_model(train_x, train_y, test_x, test_y)
                predict = self.classifier.predict(test_x)

                # metrics.accuracy_score(predict,test_y)
                reward = 0  # = 1.0 / metrics.mean_squared_error(test_y, predict)

                self.get_shap_value(train_x, test_x)
            elif self.classifier_name == 'cox':
                reward = self.calcualte_reward(solution, train=train)

        return reward

        # results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='accuracy')
    def calcualte_reward(self, solution, train=True):

        sol_list = Solution.sol_to_list(solution)

        smallgene = pd.DataFrame.copy(pd.concat([self.reward_df.iloc[:, sol_list], self.reward_df['Treatment'],
                               self.reward_df['time'], self.reward_df['event']], axis=1))

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
                                                    scoring_method="concordance_index",seed=self.config['seed'])
            print('C-index(cross-validation) = ', np.mean(cox_cv_result))
            cox_cv_result = k_fold_cross_validation(cox2, smallgene[f], duration_col='time', event_col='event', k=5,
                                                    scoring_method="concordance_index",seed=self.config['seed'])
            print('C-index(cross-validation) = ', np.mean(cox_cv_result))


    def get_shap_value(self, train_x, test_x):
        '''
        Agent의 최종 solution을 이용하여 모델을 학습하고 validation 데이터 셋에 shap value를 시각화하여 보여줌.
        :param test_x: 테스트용 데이터
        :param train_x: 학습용 데이터
        :return: 없음.
        '''
        # 그래프 초기화
        if self.classifier_name == 'linear':

            shap.initjs()

            ex = shap.KernelExplainer(self.classifier.predict, train_x)

            # 첫번째 test dataset 하나에 대해서 shap value를 적용하여 시각화
            #shap_values = ex.shap_values(test_x[0, :])
            #shap.force_plot(ex.expected_value, shap_values, test_x[0, :])

            # 전체 검증 데이터 셋에 대해서 적용
            shap_values = ex.shap_values(test_x)

            test_x_df = pd.DataFrame(test_x, columns=self.data.columns[:-1])
            shap.summary_plot(shap_values, test_x_df, max_display=30)
            plt.savefig('./linear_regression_result.jpg')

            #explainer = shap.Explainer(self.classifier)
            #shap_values = explainer(test_x[0,:])

            # visualize the first prediction's explanation
            #shap.plots.waterfall(shap_values[0])
            #plt.savefig('./xgboost_waterfall_result.jpg')
        elif self.classifier_name == 'deep':
            shap.initjs()

            ex = shap.DeepExplainer(self.classifier())
            shap_value = ex.shap_values(test_x)
            shap.summary_plot(shap_value, test_x, show=False)

            plt.savefig('shap.png')


    def train_model(self,solution, total_x, total_y):
        criterion = nn.MSELoss()

        #optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.1)

        n_epochs = 15
        kfold = KFold(n_splits=self.cv_n_split, random_state=0, shuffle=True)
        total_valid_loss=[]
        for cv_ind,(train_index, validate_index) in enumerate(kfold.split(total_x)):
            self.classifier = Model(sum(solution) + 10)
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

            x_train, x_validate = total_x[train_index], total_x[validate_index]
            y_train, y_validate = total_y[train_index], total_y[validate_index]

            train_inputs = (torch.from_numpy(x_train)).float()
            train_targets = (torch.from_numpy(y_train)).float()

            val_inputs = (torch.from_numpy(x_validate)).float()
            val_targets = (torch.from_numpy(y_validate)).float()

            train_dataset = TensorDataset(train_inputs, train_targets)
            valid_dataset = TensorDataset(val_inputs, val_targets)

            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            best_loss=10000.0
            # train
            for epoch in range(n_epochs):
                train_losses = AverageMeter('Loss', ':.4e')
                progress = ProgressMeter(
                    len(train_loader),
                    [train_losses],
                    prefix="Epoch: [{}]".format(epoch))
                self.classifier.train()

                for i,(input, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    input = Variable(input)
                    target = Variable(target)

                    predict = self.classifier(input)
                    loss = criterion(predict, target)

                    loss.backward()
                    optimizer.step()
                    train_losses.update(loss.item(), input.size(0))
                    # if i % 2 == 0:
                    #     progress.display(i)
                # print("[{} epoch] Train Loss {:.4f}".format(i,losses.avg))

                # evaluate
                eval_losses = AverageMeter('Loss', ':.4e')

                self.classifier.eval()
                for input, target in valid_loader:
                    with torch.no_grad():
                        input = Variable(input)
                        target = Variable(target)

                        predict = self.classifier(input)
                        loss = criterion(predict, target)
                    eval_losses.update(loss.item(), input.size(0))
                # if epoch%5==0:
                #     print("cv {} [{} epoch] Eval Loss {:.4f}".format(cv_ind,epoch,eval_losses.avg))
                if best_loss>eval_losses.avg:
                    best_loss=eval_losses.avg
            print('\r[{}/{} cv] best eval loss {:.4f}'.format(cv_ind+1,self.cv_n_split,best_loss),end='')
            total_valid_loss.append(best_loss)
        return_loss=torch.tensor(total_valid_loss).mean()
        print(" RETURN Loss: {:.4f}".format(return_loss.item()))
        return return_loss

    def train_best_solutoin_model(self, train_x, train_y, test_x, test_y):
        train_dataset = TensorDataset(train_x, train_y)
        valid_dataset = TensorDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        best_loss = 10000.0

        self.classifier = Model(train_x.shape[1])#sum(solution) + 10)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        # train
        for epoch in range(15):
            train_losses = AverageMeter('Loss', ':.4e')
            progress = ProgressMeter(
                len(train_loader),
                [train_losses],
                prefix="Epoch: [{}]".format(epoch))
            self.classifier.train()

            for i, (input, target) in enumerate(train_loader):
                optimizer.zero_grad()
                input = Variable(input)
                target = Variable(target)

                predict = self.classifier(input)
                loss = criterion(predict, target)

                loss.backward()
                optimizer.step()
                train_losses.update(loss.item(), input.size(0))

            # evaluate
            eval_losses = AverageMeter('Loss', ':.4e')

            self.classifier.eval()
            for input, target in valid_loader:
                with torch.no_grad():
                    input = Variable(input)
                    target = Variable(target)

                    predict = self.classifier(input)
                    loss = criterion(predict, target)
                eval_losses.update(loss.item(), input.size(0))
            if epoch % 5 == 0:
                print("[{} epoch] Eval Loss {:.4f}".format(epoch, eval_losses.avg))
            if best_loss > eval_losses.avg:
                best_loss = eval_losses.avg

        print(best_loss)

        return best_loss
