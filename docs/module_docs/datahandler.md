# DataHandler

## Table of Contents

* [datahandler.data\_feature\_distribution.gaussian](#datahandler.data_feature_distribution.gaussian)
  * [GaussianNoiseTransform](#datahandler.data_feature_distribution.gaussian.GaussianNoiseTransform)
* [datahandler.data\_feature\_distribution](#datahandler.data_feature_distribution)
* [datahandler.data\_feature\_distribution.data\_feature\_distribution](#datahandler.data_feature_distribution.data_feature_distribution)
  * [DataFeatureDistribution](#datahandler.data_feature_distribution.data_feature_distribution.DataFeatureDistribution)
    * [apply\_feature\_skew](#datahandler.data_feature_distribution.data_feature_distribution.DataFeatureDistribution.apply_feature_skew)
* [datahandler.datahandler](#datahandler.datahandler)
  * [DataHandler](#datahandler.datahandler.DataHandler)
    * [load\_distributed\_datasets](#datahandler.datahandler.DataHandler.load_distributed_datasets)
    * [get\_classes](#datahandler.datahandler.DataHandler.get_classes)
    * [split\_and\_transform\_data](#datahandler.datahandler.DataHandler.split_and_transform_data)
    * [distribute\_data](#datahandler.datahandler.DataHandler.distribute_data)
    * [load\_existing\_distribution](#datahandler.datahandler.DataHandler.load_existing_distribution)
    * [generate\_transforms](#datahandler.datahandler.DataHandler.generate_transforms)
* [datahandler.data\_label\_distribution.uniform](#datahandler.data_label_distribution.uniform)
  * [Uniform](#datahandler.data_label_distribution.uniform.Uniform)
    * [get\_label\_distribution](#datahandler.data_label_distribution.uniform.Uniform.get_label_distribution)
* [datahandler.data\_label\_distribution.data\_label\_distribution](#datahandler.data_label_distribution.data_label_distribution)
  * [DataLabelDistribution](#datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution)
    * [get\_label\_distribution](#datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution.get_label_distribution)
* [datahandler.data\_label\_distribution](#datahandler.data_label_distribution)
* [datahandler.data\_label\_distribution.discrete](#datahandler.data_label_distribution.discrete)
  * [Discrete](#datahandler.data_label_distribution.discrete.Discrete)
    * [get\_label\_distribution](#datahandler.data_label_distribution.discrete.Discrete.get_label_distribution)
* [datahandler.data\_label\_distribution.dirichlet](#datahandler.data_label_distribution.dirichlet)
  * [Dirichlet](#datahandler.data_label_distribution.dirichlet.Dirichlet)
    * [get\_label\_distribution](#datahandler.data_label_distribution.dirichlet.Dirichlet.get_label_distribution)
* [datahandler.mnist](#datahandler.mnist)
  * [MNISTDataHandler](#datahandler.mnist.MNISTDataHandler)
    * [load\_distributed\_datasets](#datahandler.mnist.MNISTDataHandler.load_distributed_datasets)
    * [get\_classes](#datahandler.mnist.MNISTDataHandler.get_classes)
* [datahandler.cifar10](#datahandler.cifar10)
  * [Cifar10DataHandler](#datahandler.cifar10.Cifar10DataHandler)
    * [load\_distributed\_datasets](#datahandler.cifar10.Cifar10DataHandler.load_distributed_datasets)
    * [get\_classes](#datahandler.cifar10.Cifar10DataHandler.get_classes)
* [datahandler.data\_quantity\_distribution.uniform](#datahandler.data_quantity_distribution.uniform)
  * [Uniform](#datahandler.data_quantity_distribution.uniform.Uniform)
    * [get\_partition\_sizes](#datahandler.data_quantity_distribution.uniform.Uniform.get_partition_sizes)
* [datahandler.data\_quantity\_distribution.data\_quantity\_distribution](#datahandler.data_quantity_distribution.data_quantity_distribution)
  * [DataQuantityDistribution](#datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution)
    * [get\_partition\_sizes](#datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution.get_partition_sizes)
* [datahandler.data\_quantity\_distribution](#datahandler.data_quantity_distribution)
* [datahandler.data\_quantity\_distribution.dirichlet](#datahandler.data_quantity_distribution.dirichlet)
  * [Dirichlet](#datahandler.data_quantity_distribution.dirichlet.Dirichlet)
    * [get\_partition\_sizes](#datahandler.data_quantity_distribution.dirichlet.Dirichlet.get_partition_sizes)

<h1 id="datahandler.data_feature_distribution.gaussian">datahandler.data_feature_distribution.gaussian</h1>

<h2 id="datahandler.data_feature_distribution.gaussian.GaussianNoiseTransform">GaussianNoiseTransform Objects</h2>

```python
class GaussianNoiseTransform(object)
```

Add Gaussian noise to a tensor

<h1 id="datahandler.data_feature_distribution">datahandler.data_feature_distribution</h1>

This module contains methods of skewing data features

<h1 id="datahandler.data_feature_distribution.data_feature_distribution">datahandler.data_feature_distribution.data_feature_distribution</h1>

DataFeatureDistribution is an abstract class that defines the interface for any implemented data feature distributions

<h2 id="datahandler.data_feature_distribution.data_feature_distribution.DataFeatureDistribution">DataFeatureDistribution Objects</h2>

```python
class DataFeatureDistribution(ABC)
```

DataFeatureDistribution is an abstract class that defines the interface for any implemented data feature distributions

<h4 id="datahandler.data_feature_distribution.data_feature_distribution.DataFeatureDistribution.apply_feature_skew">apply_feature_skew</h4>

```python
def apply_feature_skew(datahandler)
```

Applies the feature skew to the data

<h1 id="datahandler.datahandler">datahandler.datahandler</h1>

This contains the abstract data handler that defines the interface for any implemented
data handlers and provides some universal methods

<h2 id="datahandler.datahandler.DataHandler">DataHandler Objects</h2>

```python
class DataHandler(ABC)
```

DataHandler is an abstract class that defines the interface for any implemented data handlers

<h4 id="datahandler.datahandler.DataHandler.load_distributed_datasets">load_distributed_datasets</h4>

```python
@abstractmethod
def load_distributed_datasets()
```

Called to load the dataset

<h4 id="datahandler.datahandler.DataHandler.get_classes">get_classes</h4>

```python
@abstractmethod
def get_classes()
```

Returns the classes of the dataset

<h4 id="datahandler.datahandler.DataHandler.split_and_transform_data">split_and_transform_data</h4>

```python
def split_and_transform_data(testset, trainset)
```

Split the data into partitions and create DataLoaders

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

**Returns**:

testloader, trainloaders, valloaders

<h4 id="datahandler.datahandler.DataHandler.distribute_data">distribute_data</h4>

```python
def distribute_data(label_distribution, partition_sizes, trainset)
```

Distribute the data according to the label distribution and partition sizes

**Arguments**:

- `label_distribution`: np.array of shape (NUM_CLIENTS, NUM_CLASSES)
- `partition_sizes`: np.array of shape (NUM_CLIENTS)
- `trainset`: torch.utils.data.Dataset

**Returns**:

list of torch.utils.data.Subset

<h4 id="datahandler.datahandler.DataHandler.load_existing_distribution">load_existing_distribution</h4>

```python
def load_existing_distribution(trainset)
```

Load an existing data distribution from a file

**Arguments**:

- `trainset`: torch.utils.data.Dataset

**Returns**:

List of torch.utils.data.Subset

<h4 id="datahandler.datahandler.DataHandler.generate_transforms">generate_transforms</h4>

```python
def generate_transforms(custom_transforms=None)
```

Generate the transforms for the dataset

Custom transforms are applied after a tensor was created and before normalization and feature skewing

**Arguments**:

- `custom_transforms`: List of custom transforms

**Returns**:

Composed transforms

<h1 id="datahandler.data_label_distribution.uniform">datahandler.data_label_distribution.uniform</h1>

Uniform distribution of labels

<h2 id="datahandler.data_label_distribution.uniform.Uniform">Uniform Objects</h2>

```python
class Uniform(DataLabelDistribution)
```

Uniform distribution of labels

<h4 id="datahandler.data_label_distribution.uniform.Uniform.get_label_distribution">get_label_distribution</h4>

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

Uses uniform distribution to (not-)skew the data label distribution

**Returns**:

label_distribution

<h1 id="datahandler.data_label_distribution.data_label_distribution">datahandler.data_label_distribution.data_label_distribution</h1>

DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions

<h2 id="datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution">DataLabelDistribution Objects</h2>

```python
class DataLabelDistribution(ABC)
```

DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions

<h4 id="datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution.get_label_distribution">get_label_distribution</h4>

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

<h1 id="datahandler.data_label_distribution">datahandler.data_label_distribution</h1>

This module contains methods of skewing data labels

<h1 id="datahandler.data_label_distribution.discrete">datahandler.data_label_distribution.discrete</h1>

Discrete data label distribution

<h2 id="datahandler.data_label_distribution.discrete.Discrete">Discrete Objects</h2>

```python
class Discrete(DataLabelDistribution)
```

Discrete data label distribution

<h4 id="datahandler.data_label_distribution.discrete.Discrete.get_label_distribution">get_label_distribution</h4>

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension no_clients, no_classes

Allows each client to have only a subset of the classes

**Returns**:

label_distribution

<h1 id="datahandler.data_label_distribution.dirichlet">datahandler.data_label_distribution.dirichlet</h1>

Dirichlet distribution for data label distribution

<h2 id="datahandler.data_label_distribution.dirichlet.Dirichlet">Dirichlet Objects</h2>

```python
class Dirichlet(DataLabelDistribution)
```

Dirichlet distribution for data label distribution

<h4 id="datahandler.data_label_distribution.dirichlet.Dirichlet.get_label_distribution">get_label_distribution</h4>

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

Uses a dirichlet distribution to skew the data label distribution

**Returns**:

label_distribution

<h1 id="datahandler.mnist">datahandler.mnist</h1>

MNIST data handler
LeCun, Yann, Corinna Cortes, and C. J. Burges. n.d.
“MNIST Handwritten Digit Database.”
ATT Labs [Online]. Available: Http://yann. Lecun. Com/exdb/mnist.

<h2 id="datahandler.mnist.MNISTDataHandler">MNISTDataHandler Objects</h2>

```python
class MNISTDataHandler(DataHandler)
```

<h4 id="datahandler.mnist.MNISTDataHandler.load_distributed_datasets">load_distributed_datasets</h4>

```python
def load_distributed_datasets()
```

Load the MNIST dataset and divide it into partitions


<h4 id="datahandler.mnist.MNISTDataHandler.get_classes">get_classes</h4>

```python
def get_classes()
```

Returns the classes of the dataset

**Returns**:

List of classes

<h1 id="datahandler.cifar10">datahandler.cifar10</h1>

CIFAR-10 data handler
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015.
“Deep Residual Learning for Image Recognition.”
arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.

<h2 id="datahandler.cifar10.Cifar10DataHandler">Cifar10DataHandler Objects</h2>

```python
class Cifar10DataHandler(DataHandler)
```

Data handler for CIFAR-10

<h4 id="datahandler.cifar10.Cifar10DataHandler.load_distributed_datasets">load_distributed_datasets</h4>

```python
def load_distributed_datasets()
```

Load the CIFAR-10 dataset and divide it into partitions

**Returns**:

Train, validation and test data loaders

<h4 id="datahandler.cifar10.Cifar10DataHandler.get_classes">get_classes</h4>

```python
def get_classes()
```

Get the classes of the CIFAR-10 dataset

**Returns**:

List of classes

<h1 id="datahandler.data_quantity_distribution.uniform">datahandler.data_quantity_distribution.uniform</h1>

Uniform data quantity distribution

<h2 id="datahandler.data_quantity_distribution.uniform.Uniform">Uniform Objects</h2>

```python
class Uniform(DataQuantityDistribution)
```

Uniform data quantity distribution

<h4 id="datahandler.data_quantity_distribution.uniform.Uniform.get_partition_sizes">get_partition_sizes</h4>

```python
def get_partition_sizes(testset, trainset)
```

Returns the partition sizes as an array of dimension (no_clients)

Uses a uniform distribution to (not-)skew the data quantities

**Arguments**:

- `testset`: test dataset
- `trainset`: train dataset

<h1 id="datahandler.data_quantity_distribution.data_quantity_distribution">datahandler.data_quantity_distribution.data_quantity_distribution</h1>

This class contains the abstract class DataQuantityDistribution which
is used to for all implemented data quantity distributions

<h2 id="datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution">DataQuantityDistribution Objects</h2>

```python
class DataQuantityDistribution(ABC)
```

DataQuantityDistribution is an abstract class that defines the
interface for any implemented data quantity distributions

<h4 id="datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution.get_partition_sizes">get_partition_sizes</h4>

```python
def get_partition_sizes(testset, trainset)
```

Returns the number of samples to be allocated to every client

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

<h1 id="datahandler.data_quantity_distribution">datahandler.data_quantity_distribution</h1>

This module contains the classes for skewing data quantity distributions

<h1 id="datahandler.data_quantity_distribution.dirichlet">datahandler.data_quantity_distribution.dirichlet</h1>

Dirichlet distribution for data quantity distribution

<h2 id="datahandler.data_quantity_distribution.dirichlet.Dirichlet">Dirichlet Objects</h2>

```python
class Dirichlet(DataQuantityDistribution)
```

<h4 id="datahandler.data_quantity_distribution.dirichlet.Dirichlet.get_partition_sizes">get_partition_sizes</h4>

```python
def get_partition_sizes(testset, trainset)
```

Returns the number of samples to be allocated to every client

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

**Returns**:

Array of size (no_clients) containing the number of samples for every client

