# Algorithm Documentation: FedCS

FedCS aims to optimize the number of clients involved in each round. It does so by greedily selecting the largest number of clients that can feasibly complete the round of training.
This reduces the number of clients that participate, but fail to contribute to the global model in each round.

It is based on the paper 
```
Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge.
Nishio, Takayuki, and Ryo Yonetani. 2018. arXiv [cs.NI]. arXiv. http://arxiv.org/abs/1804.08333.
```
The algorithm may be selected by choosing `FedCS` as the algorithm in the config file.

It requires the following parameters:

| Key             | Description                                                                                           | Example Value     |
|-----------------|-------------------------------------------------------------------------------------------------------|-------------------|
| pre_sampling    | Describes the percentage of the total available client to select for participation in client selection | 0.2               |
| fixed_client_no | Whether the client number should be fixed or not (useful for comparison)                              | `True` or `False` |
| c_clients       | Describes the share of all clients to participate in the round (only if `fixed_client_no` is `True`)  | 0.2               |