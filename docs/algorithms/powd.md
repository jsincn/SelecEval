# Algorithm Documentation: Pow-D

The PowD client selection strategy proposed by Cho et. al. in 2020 is based on sampling
the clients with the highest local loss. Based on their convergence analysis this results in
an increased convergence rate compared to unbiased client selection. In their own testing,
they achieved 3× higher convergence speed as well as up to 10% higher accuracy compared
to random selection. We elected to use the computation efficient variant $\pi_{cpow-d}$
which uses a randomly sampled subset to estimate the client loss. For this, we used the
client’s validation set, as this is randomly sampled from a client’s assigned training samples.

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