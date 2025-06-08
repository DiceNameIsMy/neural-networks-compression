# Neural Architecture Search for Neural Network Compression with NSGA-II algorithm

## Abstract

Printed electronics have significant operational constraints, making even the simplest full precision neural networks used in typical computers not programmable into them. This work is concerned with feasibility of encoding neural networks (NNs) into printed electronics (PE) through compression techniques. 
  
It evaluates several neural network compression techniques including binarized networks, ternary weight networks, and n-bit quantization. Two complementary experiments were conducted: first, compare compression techniques on well known architectures (LeNet5 and VGG7), and second, using neural architecture search explore optimal architecture compression combinations across the accuracy-complexity trade-off space. Neural architecture search (NAS) was applied to multi-layer perceptron (MLP) and convolutional neural network (CNN) architectures across multiple datasets, generating Pareto-optimal solutions that balance accuracy and computational complexity for printed electronics constraints. 

A functioning, configurable, and extendable program for NAS is presented alongside the results of the aforementioned experiments.

## Installation

If your machine will run in CPU (It does not have CUDA), run this:
```sh
bash setup.sh
```

If it will use CUDA, run the following. You may need to use a different CUDA_PLATFORM_URL. Visit [here](https://pytorch.org/get-started/locally/) to learn more. Use a value from `--index-url` argument.
```sh
COMPUTE_PLATFORM_URL=https://download.pytorch.org/whl/cu118 bash ./setup.sh
```

Activate the virtual enviroment in your terminal:
```sh
source venv/bin/activate
```

You may now use the cli.

## Example usage

### Run experiment 1

```sh
python src/main.py -l debug nas -d mini-cifar10 --batch-size 100 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/mini-cifar10
```

To make plots from the results, see [src/notebooks/plot.ipynb](src/notebooks/plot.ipynb).

### Run experiment 2

```sh
python src/main.py -l debug nas -d vertebral --batch-size 32 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/vertebral
```

```sh
python src/main.py -l debug nas -d mini-mnist --batch-size 50 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/mini-mnist
```

To make plots from the results, see [src/notebooks/plot.ipynb](src/notebooks/plot.ipynb).

