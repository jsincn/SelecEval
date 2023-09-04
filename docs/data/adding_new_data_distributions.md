The process for adding new data distributions is as follows and identical for feature, label and quantity skews:

1. Inside the ```seleceval/datahandler``` directory, select the appropriate subdirectory for feature, label or quantity distributions
2. Create a new python file for your distribution and inherit from the appropriate base class
3. Implement the appropriate methods
4. Add the new distribution to the ```__init__.py``` file in the same directory
5. Add the new distribution to appropriate ```seleceval/util/config_parameters``` file, including any relevant parameters

You can use the existing distributions as a template for your own.