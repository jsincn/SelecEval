# Algorithm Documentation: Pow-D

Pow-D aims to optimize the number of clients involved in each round by focusing on the clients with the highest loss 
on their individual validation set.

It is based on the paper 
```
Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies.
Cho, Yae Jee, Jianyu Wang, and Gauri Joshi. 2020. arXiv.org. https://www.semanticscholar.org/paper/e245f15bdddac514454fecf32f2a3ecb069f6dec.
```
The algorithm may be selected by choosing `PowD` as the algorithm in the config file.
As a note, the algorithm as it is implemented in this context runs the validation on the entire validation set, rather than micro batches.

It requires the following parameters:

| Key          | Description                                                                          | Example Value |
|--------------|--------------------------------------------------------------------------------------|---------------|
| pre_sampling | Describes the share of all clients that should be selected to calculate their losses | 0.4           |
| c            | Describes the share of all clients that should be selected                           | 0.2           |