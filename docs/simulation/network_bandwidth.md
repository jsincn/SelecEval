# Network Bandwidth

Reducing necessary communication between the client and the server is one of the main 
goals of federated learning. 
As such, we implemented a rudimentary network and communication simulation into SelecEval. 
For this, every client is simulated as having a particular network bandwidth. 
This network bandwidth is updated every round as devices move between mobile network cells or share their bandwidth with other devices. 
The network bandwidth is then considered during the client runs to evaluate whether the client was able to complete the round within the given time limit. It is also included during client selection.

The network bandwidth is modeled as a normal distribution with a mean of 20 Mbit/s and a standard deviation of 10 Mbit/s. These values are configurable by the user.