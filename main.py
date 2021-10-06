from fs_data import FSData

if __name__=="__main__":

    # RL
    alhpa = 0.1
    gamma = 0.99
    epsilon = 0.01

    # BSO

    flip = 5
    max_chance = 3
    bees_number = 10
    maxIterations = 10
    locIterations = 10

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
        "classifier": "lightgbm",  # reward 계산용 모델
        "treatment": 0,  # 0 = 치료 안받은 데이터, 1 = 치료 받은 데이터
        "gene_num_train": 4,  # 전처리를 통해 추출할 유전자 수
    }

    for model in ['linear_regression', 'lightgbm']:
        for treat in [0,1]:
            for gene_n in [30, 40, 50]:
                config["classifier"] = model
                config['treatment'] = treat
                config['gene_num_train'] = gene_n

                instance = FSData(typeOfAlgo,location,nbr_exec,method,test_param,param,val,alhpa,gamma,epsilon, config)
                instance.run(flip,max_chance,bees_number,maxIterations,locIterations)
