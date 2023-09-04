# ActiveFL

ActiveFL is an algorithm introduced in 2019 with the goal of selecting an optimized subset of
clients to participate in federated learning based on a value function. This value
function aims to quantify the expected usefulness of a client’s data during a training round.
In addition to sampling a subset of the clients based on their valuation, another set is selected
uniformly at random. While not explicitly mentioned in the original paper we assume that
the random portion of the sampled set aims to reduce overfitting on the same clients every
round.

It is based on the paper 
```
Goetz, Jack, Kshitiz Malik, D. Bui, Seungwhan Moon, Honglei Liu, and Anuj Kumar. 2019. 
“Active Federated Learning.” arXiv.org. 
https://www.semanticscholar.org/paper/36b9b82b607149f160abde58db77149c6de58c01.
```
The algorithm may be selected by choosing `ActiveFL` as the algorithm in the config file.

It requires the following parameters:

| Key    | Description                                                    | Example Value |
|--------|----------------------------------------------------------------|---------------|
| c      | Describes the share of all clients to participate in the round | 0.2           |
| alpha1 | Percentage of lowest score clients to disqualify               | 0.75          |
| alpha2 | Parameter for selection probability                            | 0.01          |
| alpha3 | Percentage of clients to select randomly                       | 0.1           |