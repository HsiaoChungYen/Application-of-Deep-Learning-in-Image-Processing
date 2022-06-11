# Detailed Explanation of AlexNet Network Structure and Model Construction

AlexNet is the champion network of the 2012 ILSVRC 2012 (ImageNet Large Scale Visual Recognition Challenge) competition, and the classification accuracy has increased from 70%+ of the traditional method to 80%+ (the traditional method has entered the bottleneck period at that time, so such a big improvement is very good of). It was designed by Hinton and his student Alex Krizhevsky. It was also after that year that deep learning began to develop rapidly. The following figure is a network structure diagram taken from the original AlexNet paper.

![image](https://github.com/HsiaoChungYen/Application-of-Deep-Learning-in-Image-Processing/blob/main/pytorch_classification/Project2_alexnet/AlexNet's_network_structure_diagram.png)

There are upper and lower parts in the figure, because the author uses two GPUs for parallel training, so the structure of the upper and lower parts is exactly the same, we can see the following parts. Here are the highlights of the network:

(1) The first use of GPU for network acceleration training.

(2) Use the ReLU activation function to replace the traditional Sigmoid activation function and Tanh activation function.

(3) Use LRN local response normalization.

(4) In the first two layers of the fully connected layer, the Dropout method is used to randomly inactivate neurons in a certain proportion to reduce overfitting.
  
Then the formula for calculating the size of the matrix after convolution or pooling is given:

   N = (W - F + 2P ) / S + 1

where W is the input image size, F is the size of the convolution or pooling kernel, P is the number of padding pixels, and S is the stride

Next, we analyze each layer of the network in detail:
  
  
**Convolution layer 1** (since 2 GPUs are used, the number of convolution kernels needs to be multiplied by 2):  
**Conv1: kernels: 48*2=96；kernel_size: 11；padding: [1, 2] ；stride: 4**

Among them, kernels indicates the number of convolution kernels, kernel_size indicates the size of the convolution, padding indicates that the parameters of the feature matrix are zero up and down, and stride indicates the stride.

Input image shape: [224, 224, 3], output feature matrix shape: [55, 55, 96]

Shape calculation: N = (W - F + 2P ) / S + 1 = [ 224 - 11 + (1 + 2)] / 4 + 1 = 55
  
  
**Max Pooling Downsampling Layer 1**  
**Maxpool1: kernel_size: 3; pading: 0; stride: 2**  
Among them, kernel_size is the size of the pooling kernel, padding represents the parameter of zero padding up, down, left and right of the feature matrix, and stride represents the stride.

The shape of the input feature matrix: [55, 55, 96], the shape of the output feature matrix: [27, 27, 96]

Shape calculation: N = (W − F + 2P ) / S + 1 = (55 - 3) / 2 + 1 = 27
  
  
**Convolutional layer 2**  
**Conv2: kernels: 128*2=256; kernel_size: 5; padding: [2, 2]; stride: 1**  
The shape of the input feature matrix: [27, 27, 96], the shape of the output feature matrix: [27, 27, 256]

Shape calculation: N = (W − F + 2P ) / S + 1 = (27 - 5 + 4) / 1 + 1 = 27
  
  
**Max Pooling Downsampling Layer 2**  
**Maxpool2: kernel_size: 3; pading: 0; stride: 2**  
The shape of the input feature matrix: [27, 27, 256], the shape of the output feature matrix: [13, 13, 256]

Shape calculation: N = (W − F + 2P ) / S + 1 = (27 - 3) / 2 + 1 = 13
  
  
**Convolutional layer 3**  
**Conv3: kernels: 192*2=384; kernel_size: 3; padding: [1, 1]; stride: 1**  
Input feature matrix shape: [13, 13, 256], output feature matrix shape: [13, 13, 384]

Shape calculation: N = (W − F + 2P ) / S + 1 = (13 - 3 + 2) / 1 + 1 = 13
  
  
**Convolutional layer 4**  
**Conv4: kernels: 192*2=384; kernel_size: 3; padding: [1, 1]; stride: 1**  
Input feature matrix shape: [13, 13, 384], output feature matrix shape: [13, 13, 384]

Shape calculation: N = (W − F + 2P ) / S + 1 = (13 - 3 + 2) / 1 + 1 = 13
  
  
**Convolutional layer 5**  
**Conv5: kernels: 128*2=256; kernel_size: 3; padding: [1, 1]; stride: 1**  
Input feature matrix shape: [13, 13, 384], output feature matrix shape: [13, 13, 256]

Shape calculation: N = (W − F + 2P ) / S + 1 = (13 - 3 + 2) / 1 + 1 = 13
  
  
**Max Pooling Downsampling Layer 3**  
**Maxpool3: kernel_size: 3 padding: 0 stride: 2**  
Input feature matrix shape: [13, 13, 256] , output feature matrix shape: [6, 6, 256]

Shape calculation: N = (W − F + 2P ) / S + 1 = (13 - 3) / 2 + 1 = 6
  
  
**Fully connected layer 1**  
**unit_size: 4096 (unit_size is the number of fully connected layer nodes, doubled for two GPUs)**  
  
**Fully connected layer 2**  
**unit_size: 4096**  
  
**Fully connected layer 3**  
**unit_size: 1000 (this layer is the output layer, the number of output nodes corresponds to the number of categories of your classification task)**  
  
  
Finally a summary table of all layer parameters is given:  
| Layer Name | Kernel Size | Kernel Num | Padding | Stride |
|------------|-------------|------------|---------|--------|
| Conv1      |11           |96          |[1, 2]   |4       |
| Maxpool1   |3            |None        |0        |2       |
| Conv2      |5            |256         |[2, 2]   |1       |
| Maxpool2   |3            |None        |0        |2       |
| Conv3      |3            |384         |[1, 1]   |1       |
| Conv4      |3            |384         |[1, 1]   |1       |
| Conv5      |3            |256         |[1, 1]   |1       |
| Maxpool3   |3            |None        |0        |2       |
| FC1        |4096         |None        |None     |None    |
| FC2        |4096         |None        |None     |None    |
| FC3        |1000         |None        |None     |None    |
  

