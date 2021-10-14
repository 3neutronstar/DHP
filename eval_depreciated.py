# -*- coding: utf-8 -*-
from Dataloader import Dataloader
from sklearn.neighbors import KNeighborsClassifier
from solution import Solution
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation


dl=Dataloader()
gene=dl.get_k_gene(60)
survival_time=dl.get_survival_time()
event=dl.get_event()
treatment=dl.get_treatment()
clinic_var=dl.get_clinic_var()
top10gene=dl.get_top10()

# solution = ['G292', 'G88', 'G193', 'G6', 'G221', 'G35', 'G136', 'G285', 'G293', 'G148', 'G165', 'G133', 'G258', 'G73', 'G278', 'G158', 'G36', 'G141', 'G128', 'G103', 'G49', 'G283', 'G8', 'G95', 'G122', 'G235', 'G251']
# solution=[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
solution=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
sol_list = Solution.sol_to_list(solution)
all_df = pd.concat((gene, clinic_var, survival_time, treatment, event), axis=1)
this_df=pd.concat([all_df.iloc[:, sol_list],clinic_var,top10gene, all_df['Treatment'],
                all_df['time'], all_df['event']], axis=1).dropna(axis=0)
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
print('Top20 features',sorted_indices[:20])
cox_cv_result = k_fold_cross_validation(cox1, smallgene[t], duration_col='time', event_col='event', k=5,
                                        scoring_method="concordance_index")
print('C-index(cross-validation) = ', np.mean(cox_cv_result))
cox_cv_result = k_fold_cross_validation(cox2, smallgene[f], duration_col='time', event_col='event', k=5,
                                        scoring_method="concordance_index")
print('C-index(cross-validation) = ', np.mean(cox_cv_result))