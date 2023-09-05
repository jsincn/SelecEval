# Example Descriptions

The examples folder contains a few examples for running SelecEval. The examples are provided as .json files and can be run using the following command:
Runtime is provided as a rough estimate for a single run on a AMD Ryzen 9 3900X (24) @ 3.800GHz . 
If you run into memory issues you can try to adjust the share of resources per simulated client in the .json file.
Alternatively you may disable CUDA acceleration.
```bash
python -m seleceval.run example_xxx.json
```

The examples are provided as follows:

| Example        | Description                                                                                                                                                        | Runtime  |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| example_1.json | A simple example with 10 clients and 2 rounds of training on the MNIST dataset. Acceleration using CUDA is disabled. Only Random Selection and FedCS are included. | < 3 min  |
| example_2.json | A simple example with 20 clients running all 5 algorithms on the MNIST dataset.                                                                                    | < 10 min |