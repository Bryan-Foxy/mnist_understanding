# Explain of the work:

An image is simply an array (tensor) of $n$ dimension.

![2](https://github.com/Bryan-Foxy/mnist_understanding/assets/67081376/1207137b-aaa7-4eb7-a5d2-e67a7531a49e)

In our case it is a matrix (2-dimensional tensor) because it is black on white.

The work here consists of better understanding the Dense Neural Network (Or Multi Layer Perceptron (MLP), these are synonyms, a DNN and composed of several perceptrons connected together) and CNN architectures as well as the behavior of Adaptive Moment Estimation ( Adam) and Stochastic Gradient Descent(SGD)


## Architecture of the model use:
For the DNN architecture, we have:

- An input layer with 784 neurons (corresponding to the size of the image, 28x28x1).
- A hidden layer with 512 neurons.
- An output layer with 10 neurons, representing each class (0 to 9).

ReLU activation function was applied to the output of each layer.

For the CNN architecture:

- A convolutional layer producing an output with 6 channels (feature map) and a kernel size of (5,5).
- A Maxpooling layer with a size of (2,2). Maxpooling is a pooling operation that calculates the maximum value for patches of a feature map, creating a downsampled (pooled) feature map.
- A second convolutional layer with a kernel size of (5,5) returning a feature map with 8 channels.
- A third convolutional layer with a kernel size of (5,5) returning a feature map with 12 channels.

The feature map is the output of a convolutional layer. Convolution is an operation that extracts important information from an image.

In summary:

**DNN Architecture:**
- Input Layer: 784 neurons
- Hidden Layer: 512 neurons (ReLU activation)
- Output Layer: 10 neurons (for each class, with ReLU activation)

Trainable parameters of DNN: 669706

**CNN Architecture:**
- Convolutional Layer 1: 6 channels, kernel size (5,5)
- Maxpooling Layer: Size (2,2)
- Convolutional Layer 2: 8 channels, kernel size (5,5)
- Convolutional Layer 3: 12 channels, kernel size (5,5)

Trainable parameters of CNN: 370378

We can already conclude that convolutional neural networks require less computation than DNNs. This is explained by the fact that what enters as input in the dense part is a vectorized extraction of the original image whereas in the case of DNN for image classification we have as input the original vectorized image

## Results:

![arhitecture  comparaison](https://github.com/Bryan-Foxy/mnist_understanding/assets/67081376/3038021a-380d-4a74-a809-be79ecdc2169)

This is the visualization of the accuracy of DNN and CNN models and using the different optimizers

Adam Converges very quickly but appears quite unstable
SGD is more cautious and more stable

Maybe using Adam and SGD we should have different learning_rates.
Perhaps Adam is showing instability because of a high learning_rate, perhaps we should reduce the latter for him.
Could the choice of learning_rate depend on the optimizer? (For my part yes)

![loss comparaison](https://github.com/Bryan-Foxy/mnist_understanding/assets/67081376/14b53728-893e-4e5c-9e22-29e7feccc30b)


Preview in tensorboard:

Tensorboard is a web application written in Python that allows you to view the progress of your calculations in real time.

![tensorboard](https://github.com/Bryan-Foxy/mnist_understanding/assets/67081376/ae9e3903-5bf6-4023-8a8a-9d6e6d8bdcbf)


We can compare for ourselves:

| Model              | Optimizer | Accuracy | Test Loss |
|--------------------|-----------|----------|-----------|
| DNN with Adam      | Adam      | 0.9781   | 0.0797    |
| DNN with SGD       | SGD       | 0.9824   | 0.0623    |
| CNN with Adam      | Adam      | 0.9762   | 0.0774    |
| CNN with SGD       | SGD       | 0.9865   | 0.0433    |


Thank you

By FOZAME ENDEZOUMOU ARMAND BRYAN

