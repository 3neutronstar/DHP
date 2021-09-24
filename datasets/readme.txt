Data Set
Clinical_Variable.csv
개인의 임상 정보를 담은 테이블. 10개의 임상변수가 존재하며
개인의 임상정보는 0~9까지의 수치로 대체한다.

Genetic_alterations.csv
유전자의 변형이 일어났는지를 표시한 테이블.
G1~G300의 column이 존재하며 변형이 일어나지 않았으면 0, 변형이 일어났으면 1로 표현한다. 

Survival_time_event.csv
생존기간과 사망여부를 표시한 테이블
time, event의 column이 존재하며
event가 1일경우 time은 사망시 생존기간을 의미하며
event가 0일경우 time은 현재까지의 생존기간을 의미한다.

Treatment.csv
하나뿐인 치료를 하였는지(1) 하지 아니하였는지(0)를 기록한 테이블