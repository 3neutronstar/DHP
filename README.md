# DHP(Digital Health Hackathon 2021, Korea - AI Track)
- Awarded Dean's Award(1st prize) by CCEI(서울창조경제혁신센터), Naver Care
## Problem
- Reinforcement learning based gene selection that is effective for cancer treatment
- Feature selection problem

## How to deal with
### Pipeline

![PipelineImage](https://user-images.githubusercontent.com/59332148/137326198-21668670-a0cb-4968-b3db-dcce5deddfe8.png)

#### Filter Method (Cox regression)

Filter the feature subset that is expected to have a correlation between mutant gene information and survival period with p-value after the cox regression </br>

#### Wrapper Method (QBSO-FS + Cox regression)

With feature subset, QBSO-FS,one of wrapper methods, is used to suggest top 10 candidate mutant gene. </br>
Compared to the original QBSO-FS, cox regression parameters that influence the positive treatment effect compose the reward function. </br>

- Markov Decision Process (MDP)

State: Feature subset that uses for cox regression in the subset </br>
Action: Flipping whether use the feature or not </br>
Reward: ![image](https://user-images.githubusercontent.com/59332148/137331317-d4fcff57-3efa-4c5a-b32a-068516be8e33.png)  </br>

## Analysis of Examples

- The effect of G88 gene mutation
G88 is higher effect in cox_treat, less effect in cox_notreat. --> If G88 is mutant gene, cancer treatment has positive effect.
![image](https://user-images.githubusercontent.com/59332148/137347652-1035aaa5-8914-48ad-b83a-d0703ef86a63.png)

## Reference

[Cox Regression]https://www.jstor.org/stable/pdf/2532940.pdf </br>
[QBSO-FS]https://link.springer.com/chapter/10.1007/978-3-030-20518-8_65 </br>

## Code Reference

[QBSO-FS]https://github.com/amineremache/qbso-fs

### Contributor

Minkoo Kang (Leader) </br>
Minsoo Kang </br>
Dongjin Kim </br>
[KIST-KDST]https://kdst.re.kr
