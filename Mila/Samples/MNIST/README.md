# MNIST Handwritten Digit Classification

This sample demonstrates training a feed-forward neural network on the MNIST dataset using the Mila deep learning framework.

## Overview

The MNIST classifier implements a simple three-layer fully connected neural network architecture:

```
Input (784) ? Linear (128) ? GELU ? Linear (64) ? GELU ? Linear (10) ? Output
```

The model classifies 28×28 grayscale images of handwritten digits (0-9) into their corresponding classes.

## Features

- **Device-agnostic training**: Supports both CPU and CUDA execution
- **Mixed precision**: Configurable tensor precision (FP32, FP16, BF16)
- **AdamW optimizer**: Modern optimization with weight decay
- **Data augmentation**: Built-in shuffling for training data
- **Automatic batching**: Efficient mini-batch processing
- **Progress monitoring**: Real-time training metrics and accuracy reporting

## Architecture Details

### MnistClassifier
- **Input**: 784 features (28×28 flattened images)
- **Hidden Layer 1**: 784 ? 128 neurons + GELU activation
- **Hidden Layer 2**: 128 ? 64 neurons + GELU activation
- **Output Layer**: 64 ? 10 classes (one-hot encoded)
- **Total Parameters**: ~101K trainable parameters

### Data Pipeline
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Normalization**: Pixel values scaled to [0, 1]
- **Label Encoding**: One-hot encoding for classification
- **Memory**: Supports both CPU and CUDA pinned memory for efficient transfers

## Prerequisites

### Dataset

Download the MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/):

```
# Download these four files:
train-images.idx3-ubyte.gz  # Training set images
train-labels.idx1-ubyte.gz  # Training set labels
t10k-images.idx3-ubyte.gz   # Test set images
t10k-labels.idx1-ubyte.gz   # Test set labels
```

Extract them to your data directory:

```
Data/
??? DataSets/
    ??? Mnist/
        ??? train-images.idx3-ubyte
        ??? train-labels.idx1-ubyte
        ??? t10k-images.idx3-ubyte
        ??? t10k-labels.idx1-ubyte
```

### System Requirements

- **CMake**: >= 3.28
- **C++ Compiler**: C++23 support required
- **CUDA** (optional): CUDA Toolkit for GPU acceleration
- **Mila Framework**: Built and installed

## Building

The MNIST sample is built as part of the Mila samples:

```bash
# From repository root
mkdir build && cd build
cmake ..
cmake --build . --target MnistApp
```

The executable will be created in `build/MnistApp[.exe]`.

## Usage

### Basic Training

```bash
# Train with default settings (CUDA, batch size 128, 5 epochs)
./MnistApp

# Specify data directory
./MnistApp --data-dir ./path/to/mnist/data
```

### Command-Line Options

```
Usage: MnistApp [options]

Options:
  --data-dir <path>       Path to MNIST data directory 
                          (default: ./Data/DataSets/Mnist)
  --batch-size <int>      Batch size (default: 128)
  --epochs <int>          Number of epochs (default: 5)
  --learning-rate <float> Learning rate (default: 0.001)
  --beta1 <float>         Adam beta1 parameter (default: 0.9)
  --beta2 <float>         Adam beta2 parameter (default: 0.999)
  --weight-decay <float>  Weight decay (default: 0.01)
  --device <string>       Compute device: cpu or cuda (default: cuda)
  --precision <string>    Precision policy: auto, performance, accuracy, disabled
                          (default: auto)
  --help                  Show this help message
```

### Training Examples

```bash
# Train on CPU with higher learning rate
./MnistApp --device cpu --learning-rate 0.005

# Train for 10 epochs with larger batch size
./MnistApp --epochs 10 --batch-size 256

# Use accuracy-focused precision policy
./MnistApp --precision accuracy

# Train with custom hyperparameters
./MnistApp --batch-size 64 --learning-rate 0.0005 --weight-decay 0.001
```

## Training Output

Example training output:

```
Configuration:
  Data directory: ./Data/DataSets/Mnist
  Batch size: 128
  Epochs: 5
  Learning rate: 0.001
  Device: CUDA
  Precision policy: Auto

MNIST DataLoader initialized with 60000 samples and 468 batches on device: CUDA:0

Model built successfully!

MNIST Classifier: MnistMLP
Architecture:
  Input:   784 features (28x28 flattened)
  Layer 1: 784 -> 128 + GELU
  Layer 2: 128 -> 64 + GELU
  Output:  64 -> 10 classes
Parameters: 101770

Starting training for 5 epochs...

Epoch 1 [468/468] - Loss: 0.2156 - Accuracy: 93.75%
Epoch 1/5 - Time: 12.34s - Loss: 0.3421 - Accuracy: 90.12% 
  - Test Loss: 0.2891 - Test Accuracy: 91.55% - LR: 1.000e-03

Epoch 2 [468/468] - Loss: 0.1834 - Accuracy: 95.31%
Epoch 2/5 - Time: 11.98s - Loss: 0.2145 - Accuracy: 93.67%
  - Test Loss: 0.1923 - Test Accuracy: 94.21% - LR: 1.000e-03

...

Training complete!
```

## Code Structure

```
Samples/MNIST/
??? CMakeLists.txt              # Build configuration
??? README.md                   # This file
??? Src/
    ??? Mnist.cpp              # Main training application
    ??? MnistClassifier.ixx    # Neural network architecture
    ??? MnistDataLoader.ixx    # MNIST dataset loader
    ??? MnistConfig.ixx        # Configuration structures
```

### Key Components

#### MnistClassifier (`MnistClassifier.ixx`)
- Composite module implementing the 3-layer network
- Device-templated for CPU/CUDA execution
- Precision-templated for FP32/FP16/BF16
- Manages forward/backward passes and intermediate buffers

#### MnistDataLoader (`MnistDataLoader.ixx`)
- Reads MNIST IDX binary format
- Automatic normalization and one-hot encoding
- Batching and shuffling support
- Memory resource templated for CPU/pinned memory

#### Training Loop (`Mnist.cpp`)
- AdamW optimizer integration
- Loss computation (softmax cross-entropy)
- Progress reporting
- Test set evaluation

## Performance Tips

### GPU Optimization
- Use `--precision performance` for mixed precision training
- Increase batch size to maximize GPU utilization
- Use pinned memory for faster host-device transfers

### Training Tips
- Start with default hyperparameters
- Increase learning rate for faster convergence (with caution)
- Use weight decay for better generalization
- Monitor test accuracy to prevent overfitting

## Expected Results

With default hyperparameters:
- **Training Accuracy**: ~95-97% after 5 epochs
- **Test Accuracy**: ~94-96% after 5 epochs
- **Training Time**: ~10-15 seconds per epoch (CUDA)

## Troubleshooting

### Data File Not Found
```
Error: MNIST data directory not found: ./Data/DataSets/Mnist
```
**Solution**: Download and extract MNIST dataset to the correct directory.

### CUDA Out of Memory
```
CUDA error: out of memory
```
**Solution**: Reduce batch size using `--batch-size 64` or `--batch-size 32`.

### Low Accuracy
**Solution**: 
- Verify data files are not corrupted
- Try increasing epochs: `--epochs 10`
- Adjust learning rate: `--learning-rate 0.0005`

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Mila Framework Documentation](../../docs/README.md)
- [AdamW Optimizer Paper](https://arxiv.org/abs/1711.05101)
- [GELU Activation Paper](https://arxiv.org/abs/1606.08415)

## License

This sample is part of the Mila framework and follows the project license.
