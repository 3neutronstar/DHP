import pandas as pd
import os
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import numpy as np

class Dataloader(object):
    def __init__(self):
        self.data = self._load_csv()

    def _filter_method(self,data):
        '''
        cox p value기반 filtering index잡기
        '''

        cox = CoxPHFitter()

        #fitting
        total=data['total']
        gene=data['gene']
        all=pd.concat([total,gene],axis=1)
        cox.fit(all, duration_col='time', event_col='event', show_progress=True)
        df=cox.summary
        df=df.loc[df.index.isin([f'G{i}' for i in range(1,301)])]
        i=df.sort_values(by=['-log2(p)'],ascending=False)
        index=i.index
      
        # #cross-validation
        # cox_cv_result = k_fold_cross_validation(cox, all, duration_col='time', event_col='event', k=5,seed=0)
        # print('C-index(cross-validation) = ', np.mean(cox_cv_result))
        print(list(index))
        return index

    def _load_csv(self, data_dir="datasets"):
        cur_dir = os.getcwd()
        data_dir_path = os.path.join(cur_dir, data_dir)
        data = {}
        csv_name = {
            "clinic": "Clinical_Variables.csv",
            "gene": "Genetic_alterations.csv",
            "treatment": "Treatment.csv",
            "survival": "Survival_time_event.csv",
            "total": "Total_data.csv"
        }

        for type, name in csv_name.items():
            csv_path = os.path.join(data_dir_path, name)

            data[type] = pd.read_csv(csv_path, index_col=0)

        self.gene_p_value_arg_min = self._filter_method(data)
        # self.gene_p_value_arg_min=[f'G{i}' for i in range(1,301)]
        # self.gene_p_value_arg_min = ['G75', 'G173', 'G192', 'G27', 'G202', 'G63', 'G179', 'G260', 'G65', 'G1', 'G177', 'G96',
        #                         'G93', 'G175', 'G186', 'G122', 'G145', 'G53', 'G281', 'G14', 'G154', 'G25', 'G139',
        #                         'G9', 'G272', 'G112', 'G89', 'G167', 'G200', 'G20', 'G224', 'G55', 'G257', 'G269',
        #                         'G249', 'G246', 'G210', 'G238', 'G231', 'G88', 'G190', 'G103', 'G265', 'G68', 'G262',
        #                         'G24', 'G199', 'G2', 'G125', 'G289', 'G128', 'G124', 'G230', 'G254', 'G263', 'G18',
        #                         'G150', 'G73', 'G44', 'G287', 'G64', 'G148', 'G49', 'G52', 'G130', 'G212', 'G156',
        #                         'G225', 'G271', 'G188', 'G90', 'G66', 'G236', 'G19', 'G133', 'G165', 'G227', 'G161',
        #                         'G108', 'G195', 'G250', 'G244', 'G159', 'G292', 'G208', 'G37', 'G217', 'G8', 'G129',
        #                         'G290', 'G203', 'G118', 'G300', 'G248', 'G171', 'G256', 'G70', 'G107', 'G226', 'G283',
        #                         'G149', 'G168', 'G30', 'G278', 'G115', 'G242', 'G211', 'G113', 'G60', 'G82', 'G97',
        #                         'G205', 'G126', 'G209', 'G239', 'G132', 'G17', 'G172', 'G144', 'G213', 'G215', 'G32',
        #                         'G41', 'G245', 'G261', 'G163', 'G80', 'G94', 'G234', 'G174', 'G59', 'G7', 'G74', 'G298',
        #                         'G233', 'G3', 'G232', 'G201', 'G84', 'G136', 'G99', 'G162', 'G21', 'G50', 'G134',
        #                         'G140', 'G219', 'G178', 'G38', 'G120', 'G153', 'G237', 'G189', 'G282', 'G214', 'G176',
        #                         'G61', 'G117', 'G76', 'G40', 'G276', 'G152', 'G77', 'G274', 'G160', 'G111', 'G34',
        #                         'G102', 'G155', 'G5', 'G297', 'G56', 'G295', 'G222', 'G47', 'G35', 'G11', 'G101', 'G46',
        #                         'G268', 'G137', 'G286', 'G240', 'G92', 'G43', 'G62', 'G135', 'G157', 'G299', 'G100',
        #                         'G16', 'G143', 'G218', 'G36', 'G86', 'G4', 'G264', 'G296', 'G147', 'G69', 'G72', 'G81',
        #                         'G259', 'G164', 'G141', 'G131', 'G85', 'G247', 'G26', 'G196', 'G6', 'G51', 'G71',
        #                         'G181', 'G57', 'G184', 'G13', 'G228', 'G221', 'G284', 'G294', 'G216', 'G31', 'G151',
        #                         'G67', 'G191', 'G87', 'G116', 'G252', 'G33', 'G123', 'G170', 'G54', 'G48', 'G28',
        #                         'G206', 'G104', 'G279', 'G229', 'G158', 'G79', 'G291', 'G106', 'G255', 'G110', 'G266',
        #                         'G223', 'G235', 'G198', 'G91', 'G243', 'G204', 'G12', 'G169', 'G288', 'G197', 'G23',
        #                         'G119', 'G183', 'G220', 'G251', 'G241', 'G95', 'G42', 'G267', 'G185', 'G207', 'G285',
        #                         'G105', 'G127', 'G182', 'G180', 'G146', 'G10', 'G98', 'G193', 'G58', 'G258', 'G187',
        #                         'G275', 'G166', 'G280', 'G83', 'G270', 'G277', 'G78', 'G22', 'G142', 'G15', 'G114',
        #                         'G45', 'G293', 'G253', 'G29', 'G109', 'G121', 'G138', 'G39', 'G194', 'G273']

        return data

    def get_k_gene(self, gene_select_num):
        '''
        :param gene_select_num: 필요한 유전자 갯수
        :return: 모든 환자의 유전자 정보 중 p-value가 작은 유전자 순으로 gene_select_num 갯수만큼 추출하여 반환.
        '''

        new_order_data = self.data['gene'][self.gene_p_value_arg_min[:gene_select_num]]

        return new_order_data

    def get_event(self):
        return self.data['survival']['event']

    def get_treatment(self):
        return self.data['treatment']['Treatment']
    
    def get_survival_time(self):
        return self.data['survival']['time']

    def get_clinic_var(self):
        return self.data['clinic']


if __name__ == '__main__':
    dl = Dataloader()
    print(dl.get_k_gene(300))

