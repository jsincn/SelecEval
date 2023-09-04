Adding new datasets to SelecEval can easily be done. 
The following steps are required:

1. Create a new class in the seleceval/datahandler directory that inherits from the DataHandler class.
2. Implement the ```load_distributed_datasets()``` and the ```get_classes()``` methods

The ```load_distributed_datasets()``` method returns a list of trainloaders, validationloaders and the testloader.
You may use the ```split_and_transform_data()``` method to split the data into the different loaders 
based on the existing distribution strategies.
You can also manually create the split should your dataset already contain metadata based on which you wish to allocate the date to the different clients.

The ```get_classes()``` method returns a tuple of the classes in the dataset.