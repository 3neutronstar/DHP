from Dataloader import Dataloader
from swarm import Swarm
from fs_problem import FsProblem
import pandas as pd
import os, re, time, sys
from rl import QLearning
from solution import Solution
import xlsxwriter

class FSData():

    def __init__(self,typeOfAlgo,location,nbr_exec, method, test_param, param, val,
                 alpha=None,gamma=None,epsilon=None,num_k_gene=30, config=None):

        self.typeOfAlgo = typeOfAlgo
        self.location = location + ".csv"
        self.clinical_variable_location = location + "_clinical_variable.csv"
        self.nb_exec = nbr_exec
        self.dataset_name = re.search('[A-Za-z\-]*.csv',self.location)[0].split('.')[0]
        self.dl=Dataloader()
        gene=self.dl.get_k_gene(num_k_gene)
        survival_time=self.dl.get_survival_time()
        event=self.dl.get_event()
        treatment=self.dl.get_treatment()
        clinic_var=self.dl.get_clinic_var()
        self.config=config

        self.all_df = pd.concat((gene, clinic_var, survival_time, treatment, event), axis=1)
        df=pd.concat((gene,event,treatment, clinic_var,survival_time),axis=1)
        df=df.loc[df['event']==1]
        df=df.loc[df['Treatment'] == config['treatment']]
        df.drop(columns=['Treatment', 'event'], inplace=True)

        self.clinical_variable = df.loc[:, 'Var1':'Var10']
        df.drop(columns=['Var'+str(col) for col in range(1, 11)], inplace=True)

        self.df=df
        self.classifier_name = config['classifier']

        self.ql = QLearning(len(self.df.columns),Solution.attributs_to_flip(len(self.df.columns)-1),alpha,gamma,epsilon)
        self.fsd = FsProblem(self.typeOfAlgo,self.df,self.clinical_variable,self.ql,
                             classifier=self.classifier_name, reward_df=self.all_df,config=config)
        self.classifier_name = config['classifier']

        path = os.path.join('results', 'parameters', method, test_param, param, val, self.classifier_name, self.dataset_name)
        log_dir = os.path.join(path, 'logs')
        sheet_dir = os.path.join(path, 'sheets')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sheet_dir, exist_ok=True)

        self.instance_name = self.dataset_name + '_' + str(time.strftime("%m-%d-%Y_%H-%M-%S_", time.localtime()) + self.classifier_name)
        log_filename = os.path.join(log_dir, self.instance_name)

        log_file = open(log_filename + '.txt','w+')
        # sys.stdout = log_file
        
        print("[START] Dataset " + self.dataset_name + " description \n")
        print("Shape : " + str(self.df.shape) + "\n")
        print(self.df.describe())
        print("\n[END] Dataset " + self.dataset_name + " description\n")
        print("[START] Ressources specifications\n")
        print("[END] Ressources specifications\n")

        sheet_filename = str(path + '/sheets/'+ self.instance_name )
        self.workbook = xlsxwriter.Workbook(sheet_filename + '.xlsx')
        
        self.worksheet = self.workbook.add_worksheet(self.classifier_name)
        self.worksheet.write(0,0,"Iteration")
        self.worksheet.write(0,1,"Accuracy")
        self.worksheet.write(0,2,"N_Features")
        self.worksheet.write(0,3,"Time")
        self.worksheet.write(0,4,"Top_10%_features")
        self.worksheet.write(0,5,"Size_sol_space")
    
    def run(self,flip,max_chance,bees_number,maxIterations,locIterations):
        total_time = 0
        
        for itr in range(1,self.nb_exec+1):
          print ("Execution {0}".format(str(itr)))
          self.fsd = FsProblem(self.typeOfAlgo,self.df,self.clinical_variable,self.ql,
                               classifier=self.classifier_name, reward_df=self.all_df,config=self.config)
          swarm = Swarm(self.fsd,flip,max_chance,bees_number,maxIterations,locIterations)

          t1 = time.time()
          best, best_solution = swarm.bso(self.typeOfAlgo,flip)
          t2 = time.time()

          self.fsd.evaluate(best_solution, train=False, feature_name=self.dl.gene_p_value_arg_min)
          total_time += t2-t1
          print("Time elapsed for execution {0} : {1:.2f} s\n".format(itr,t2-t1))
          self.worksheet.write(itr, 0, itr)
          self.worksheet.write(itr, 1, "{0:.2f}".format(best[0]))
          self.worksheet.write(itr, 2, best[1])
          self.worksheet.write(itr, 3, "{0:.3f}".format(t2-t1))
          self.worksheet.write(itr, 4, "{0}".format(str([j[0] for j in [i for i in swarm.best_features()]])))
          self.worksheet.write(itr, 5, len(Solution.solutions))
          
        print ("Total execution time of {0} executions \nfor dataset \"{1}\" is {2:.2f} s".format(self.nb_exec,self.dataset_name,total_time))
        self.workbook.close()
