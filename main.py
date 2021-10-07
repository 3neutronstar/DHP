from fs_data import FSData
import numpy as np
import random
if __name__=="__main__":
    # fix random seed
    seed=1
    np.random.seed(seed)
    random.seed(seed)

    # RL 

    alhpa = 0.1
    gamma = 0.99
    epsilon = 0.01

    # BSO

    flip = 3
    max_chance = 5
    bees_number = 10
    maxIterations = 20
    locIterations = 20

    #gene feature selection
    num_k_gene = 60

    # Test type

    typeOfAlgo = 1
    nbr_exec = 1
    dataset = "new"
    data_loc_path = "./datasets/"
    location = data_loc_path + dataset
    method = "qbso_simple"
    test_param = "rl"
    param = "gamma"
    val = str(locals()[param])

    config = {
        'treatment': 0, # 0이면 치료받지 않은 환자 데이터 가져오기, 1이면 치료받은 환자 데이터 가져오기
        'classifier': "cox", # linear이면 reward함수 linear regression, deep이면 MLP.
        'seed':seed
    }
    instance = FSData(typeOfAlgo,location,nbr_exec,method,test_param,param,val,alhpa,gamma,epsilon,num_k_gene, config)
    instance.run(flip,max_chance,bees_number,maxIterations,locIterations)
