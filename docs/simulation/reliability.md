# Client Reliability
Client reliability is modeled as a part of device heterogeneity. In
federated learning, the server cannot control the clients. It cannot influence when clients
disconnect or become unavailable for training. It could happen during the active training
process on the client. This is particularly important for portable devices relying on mobile
network connectivity, as they may disconnect due to challenging channel environments or
coverage gaps. To model the risk of a client dropping out, we introduce a reliability score
`r`. It is the probability of the client dropping out during every round. At the beginning of
each training round, if selected, the client fails with the given probability. If a client drops
out during one round, it will remain in the pool and may be selected again in the following
round. The client reliability is generated for each client at the beginning of the simulation
and remains constant throughout the run.

The reliability score `r` is modeled as an exponential distribution with a default rate of 10. This is configurable by the user.