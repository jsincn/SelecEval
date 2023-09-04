The Dirichlet-based distribution uses a Dirichlet distribution to determine the share of each label for every client. The distribution parameter α is
customizable by the user. Following related literature, a default value of α = 0.5 is set.

### Parameters


| Key                               | Description                                        | Example Value |
|-----------------------------------|----------------------------------------------------|---------------|
| data_label_distribution_parameter | Minimum number of samples to assign to each client | 1             |


Based on:
```
Li, Qinbin, Yiqun Diao, Quan Chen, and Bingsheng He. 2022. 
“Federated Learning on Non-IID Data Silos: An Experimental Study.” 
In 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE. 
https://doi.org/10.1109/icde53745.2022.00077.
```
```
Hsu, Tzu-Ming Harry, Hang Qi, and Matthew Brown. 2019. 
“Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.” 
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1909.06335.
```