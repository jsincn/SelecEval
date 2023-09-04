This distribution type is a simple uniform distribution. This assumes an ideal
scenario with an equal sample size for all clients. Each client is assigned a fixed percentage
of the overall dataset. The total number of samples depends on the total number of clients.
The samples are selected with replacement from the entire training set.

### Parameters

| Key                          | Description                                 | Example Value |
|------------------------------|---------------------------------------------|---------------|
| data_quantity_base_parameter | Percentage of data to assign to each client | 0.01          |