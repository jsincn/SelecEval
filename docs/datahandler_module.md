<a id="datahandler"></a>

# datahandler

This module contains the available datasets and data handlers

<a id="datahandler.data_feature_distribution"></a>

# datahandler.data\_feature\_distribution

This module contains methods of skewing data features

<a id="datahandler.datahandler"></a>

# datahandler.datahandler

This contains the abstract data handler that defines the interface for any implemented
data handlers and provides some universal methods

<a id="datahandler.datahandler.DataHandler"></a>

## DataHandler Objects

```python
class DataHandler(ABC)
```

DataHandler is an abstract class that defines the interface for any implemented data handlers

<a id="datahandler.datahandler.DataHandler.load_distributed_datasets"></a>

#### load\_distributed\_datasets

```python
@abstractmethod
def load_distributed_datasets()
```

Called to load the dataset

<a id="datahandler.datahandler.DataHandler.get_classes"></a>

#### get\_classes

```python
@abstractmethod
def get_classes()
```

Returns the classes of the dataset

<a id="datahandler.datahandler.DataHandler.split_and_transform_data"></a>

#### split\_and\_transform\_data

```python
def split_and_transform_data(testset, trainset)
```

Split the data into partitions and create DataLoaders

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

**Returns**:

testloader, trainloaders, valloaders

<a id="datahandler.datahandler.DataHandler.distribute_data"></a>

#### distribute\_data

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

<a id="datahandler.datahandler.DataHandler.load_existing_distribution"></a>

#### load\_existing\_distribution

```python
def load_existing_distribution(trainset)
```

Load an existing data distribution from a file

**Arguments**:

- `trainset`: torch.utils.data.Dataset

**Returns**:

List of torch.utils.data.Subset

<a id="datahandler.data_label_distribution.uniform"></a>

# datahandler.data\_label\_distribution.uniform

Uniform distribution of labels

<a id="datahandler.data_label_distribution.uniform.Uniform"></a>

## Uniform Objects

```python
class Uniform(DataLabelDistribution)
```

Uniform distribution of labels

<a id="datahandler.data_label_distribution.uniform.Uniform.get_label_distribution"></a>

#### get\_label\_distribution

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

Uses uniform distribution to (not-)skew the data label distribution

**Returns**:

label_distribution

<a id="datahandler.data_label_distribution.data_label_distribution"></a>

# datahandler.data\_label\_distribution.data\_label\_distribution

DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions

<a id="datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution"></a>

## DataLabelDistribution Objects

```python
class DataLabelDistribution(ABC)
```

DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions

<a id="datahandler.data_label_distribution.data_label_distribution.DataLabelDistribution.get_label_distribution"></a>

#### get\_label\_distribution

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

<a id="datahandler.data_label_distribution"></a>

# datahandler.data\_label\_distribution

This module contains methods of skewing data labels

<a id="datahandler.data_label_distribution.discrete"></a>

# datahandler.data\_label\_distribution.discrete

Discrete data label distribution

<a id="datahandler.data_label_distribution.discrete.Discrete"></a>

## Discrete Objects

```python
class Discrete(DataLabelDistribution)
```

Discrete data label distribution

<a id="datahandler.data_label_distribution.discrete.Discrete.get_label_distribution"></a>

#### get\_label\_distribution

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension no_clients, no_classes

Allows each client to have only a subset of the classes

**Returns**:

label_distribution

<a id="datahandler.data_label_distribution.dirichlet"></a>

# datahandler.data\_label\_distribution.dirichlet

Dirichlet distribution for data label distribution

<a id="datahandler.data_label_distribution.dirichlet.Dirichlet"></a>

## Dirichlet Objects

```python
class Dirichlet(DataLabelDistribution)
```

Dirichlet distribution for data label distribution

<a id="datahandler.data_label_distribution.dirichlet.Dirichlet.get_label_distribution"></a>

#### get\_label\_distribution

```python
def get_label_distribution()
```

Returns the label distribution as an array of dimension (no_clients, no_classes)

Uses a dirichlet distribution to skew the data label distribution

**Returns**:

label_distribution

<a id="datahandler.mnist"></a>

# datahandler.mnist

MNIST data handler
LeCun, Yann, Corinna Cortes, and C. J. Burges. n.d.
“MNIST Handwritten Digit Database.”
ATT Labs [Online]. Available: Http://yann. Lecun. Com/exdb/mnist.

<a id="datahandler.mnist.MNISTDataHandler"></a>

## MNISTDataHandler Objects

```python
class MNISTDataHandler(DataHandler)
```

<a id="datahandler.mnist.MNISTDataHandler.load_distributed_datasets"></a>

#### load\_distributed\_datasets

```python
def load_distributed_datasets()
```

Load the MNIST dataset and divide it into partitions


<a id="datahandler.mnist.MNISTDataHandler.get_classes"></a>

#### get\_classes

```python
def get_classes()
```

Returns the classes of the dataset

**Returns**:

List of classes

<a id="datahandler.cifar10"></a>

# datahandler.cifar10

CIFAR-10 data handler
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015.
“Deep Residual Learning for Image Recognition.”
arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.

<a id="datahandler.cifar10.Cifar10DataHandler"></a>

## Cifar10DataHandler Objects

```python
class Cifar10DataHandler(DataHandler)
```

Data handler for CIFAR-10

<a id="datahandler.cifar10.Cifar10DataHandler.load_distributed_datasets"></a>

#### load\_distributed\_datasets

```python
def load_distributed_datasets()
```

Load the CIFAR-10 dataset and divide it into partitions

**Returns**:

Train, validation and test data loaders

<a id="datahandler.cifar10.Cifar10DataHandler.get_classes"></a>

#### get\_classes

```python
def get_classes()
```

Get the classes of the CIFAR-10 dataset

**Returns**:

List of classes

<a id="datahandler.data_quantity_distribution.uniform"></a>

# datahandler.data\_quantity\_distribution.uniform

Uniform data quantity distribution

<a id="datahandler.data_quantity_distribution.uniform.Uniform"></a>

## Uniform Objects

```python
class Uniform(DataQuantityDistribution)
```

Uniform data quantity distribution

<a id="datahandler.data_quantity_distribution.uniform.Uniform.get_partition_sizes"></a>

#### get\_partition\_sizes

```python
def get_partition_sizes(testset, trainset)
```

Returns the partition sizes as an array of dimension (no_clients)

Uses a uniform distribution to (not-)skew the data quantities

**Arguments**:

- `testset`: test dataset
- `trainset`: train dataset

<a id="datahandler.data_quantity_distribution.data_quantity_distribution"></a>

# datahandler.data\_quantity\_distribution.data\_quantity\_distribution

This class contains the abstract class DataQuantityDistribution which
is used to for all implemented data quantity distributions

<a id="datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution"></a>

## DataQuantityDistribution Objects

```python
class DataQuantityDistribution(ABC)
```

DataQuantityDistribution is an abstract class that defines the
interface for any implemented data quantity distributions

<a id="datahandler.data_quantity_distribution.data_quantity_distribution.DataQuantityDistribution.get_partition_sizes"></a>

#### get\_partition\_sizes

```python
def get_partition_sizes(testset, trainset)
```

Returns the number of samples to be allocated to every client

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

<a id="datahandler.data_quantity_distribution"></a>

# datahandler.data\_quantity\_distribution

This module contains the classes for skewing data quantity distributions

<a id="datahandler.data_quantity_distribution.dirichlet"></a>

# datahandler.data\_quantity\_distribution.dirichlet

Dirichlet distribution for data quantity distribution

<a id="datahandler.data_quantity_distribution.dirichlet.Dirichlet"></a>

## Dirichlet Objects

```python
class Dirichlet(DataQuantityDistribution)
```

<a id="datahandler.data_quantity_distribution.dirichlet.Dirichlet.get_partition_sizes"></a>

#### get\_partition\_sizes

```python
def get_partition_sizes(testset, trainset)
```

Returns the number of samples to be allocated to every client

**Arguments**:

- `testset`: test dataset
- `trainset`: training dataset

**Returns**:

Array of size (no_clients) containing the number of samples for every client

