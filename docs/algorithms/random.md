# Random

The simplest selection strategy is random selection. It was first introduced along with federated learning itself in 2016. It is still the most frequently used strategy due to its
simplicity and good results. The core parameter for this strategy is c, which is the percentage of available clients to select for a particular learning round. Common values for c range
between 10% and 20%, depending on the total number of available clients.

It is based on the paper 
```
McMahan, H. Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y. Arcas. 2016. 
“Communication-Efficient Learning of Deep Networks from Decentralized Data.” 
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1602.05629.
```
The algorithm may be selected by choosing `random` as the algorithm in the config file.

It requires the following parameters:

| Key | Description                                                    | Example Value |
|-----|----------------------------------------------------------------|---------------|
| c   | Describes the share of all clients to participate in the round | 0.2           |