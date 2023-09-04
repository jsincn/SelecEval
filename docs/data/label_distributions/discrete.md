Discrete distribution is a type of label distribution. In this case,
every client is only allocated data from a subset of the available classes. This technique is
included as it is widely used in literature. However, due to the
very extreme nature of this partitioning, this may not be entirely realistic. The user may
select the number of classes assigned to each client.

### Parameters

| Key                       | Description                                  | Example Value |
|---------------------------|----------------------------------------------|---------------|
| data_label_class_quantity | Number of classes to allocate to each client | 2             |

Based on
```
McMahan, H. Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y. Arcas. 2016. 
“Communication-Efficient Learning of Deep Networks from Decentralized Data.” 
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1602.05629.
```
```
Yu, Felix X., Ankit Singh Rawat, Aditya Krishna Menon, and Sanjiv Kumar. 2020. 
“Federated Learning with Only Positive Labels.” 
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2004.10342.
```
```
Li, Qinbin, Yiqun Diao, Quan Chen, and Bingsheng He. 2022. 
“Federated Learning on Non-IID Data Silos: An Experimental Study.” 
In 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE. 
https://doi.org/10.1109/icde53745.2022.00077.
```
