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
**Conv1: kernels:48*2=96；kernel_size:11；padding:[1, 2] ；stride:4**

|Name |Quantity|
|-----|--------|
|Apple|3       |
|Egg  |12      |
