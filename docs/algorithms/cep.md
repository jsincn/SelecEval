# CEP

The client eligibility protocol for federated learning was introduced by Asad et al. in 2022. The core idea of this strategy is to select clients based on their performance in
16 4 Implementation
previous communication rounds. For this, a subset of clients is initially selected that fulfills
the performance requirements. In addition, all clients are given an initial client eligibility
score (CES) of 75. Then, based on their previous round performance, they are rewarded or
penalized. In our implementation of CEP, each client is rewarded 10 points for successful
participation and punished by 5 points if it fails due to a timeout. If a client failed the
previous five rounds in a row, they will be penalized by an additional −20 Points. Due to
different implementation constraints between the simulation used by Asad et al. and our
simulator, we could not implement the full original algorithm. However, our adjusted variant
achieves respectable results, in line with those achieved in the original paper. 

It is based on the paper 
```
Asad, Muhammad, Safa Otoum, and Saima Shaukat. 2022. 
“Resource and Heterogeneity-Aware Clients Eligibility Protocol in Federated Learning.” 
In GLOBECOM 2022 - 2022 IEEE Global Communications Conference, 1140–45.
```
The algorithm may be selected by choosing `CEP` as the algorithm in the config file.

It requires the following parameters:

| Key | Description                                                    | Example Value |
|-----|----------------------------------------------------------------|---------------|
| c   | Describes the share of all clients to participate in the round | 0.2           |