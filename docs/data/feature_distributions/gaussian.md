In a real-world federated learning setting, feature skew
could be caused by regional differences between the clients and different sensor characteristics. 
As the simulator focuses on computer vision applications, specific Gaussian noise may
be added to each client’s data to simulate this feature skew.

### Parameters
| Key                       | Description                         | Example Value |
|---------------------------|-------------------------------------|---------------|
| data_feature_skew_mu | Mean for feature skew               | 0             |
| data_feature_skew_std | Standard deviation for feature skew | 1             |