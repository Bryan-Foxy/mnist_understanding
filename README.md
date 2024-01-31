# MNIST_Understanding
![mnist](https://github.com/Bryan-Foxy/mnist_understanding/assets/67081376/776928d8-e4d8-4682-9e6d-e82f529d414d)

Here, we will work with the MNIST dataset. This project I have undertaken aims to create a Deep Neural Network (DNN) after Convolution Neural Network(CNN) model initially, evaluate it, and understand comprehensively the various parameters from a mathematical perspective.


# Setup 
We have use python and the framework pytorch 2 for this project


You need Anaconda to run the job and have the basic libraries
If you want to download the project
Download the master branch and first you must install:

```
pip install pytorch 
pip install tensorboard
```

----------------------------------------------------------------------------------------------------------------------------------------------------
# DNN Model
We will begin by building a Deep Neural Network (DNN) model and testing two very popular optimizers: Stochastic Gradient Descent (SGD) and Adam. This comparison aims to observe which optimizer converges better. Additionally, we will visualize the convergence to draw conclusions for future projects.

1. **Stochastic Gradient Descent (SGD):**
  SGD is a fundamental optimization algorithm that updates the model parameters based on the gradient of the loss function with respect to those parameters. It randomly samples a small batch of data (mini-batch) from the training set for each iteration, computes the gradient on that mini-batch, and updates the parameters in the opposite direction of the gradient. This process is repeated until convergence. SGD has a learning rate hyperparameter that determines the step size during each update.

2. **Adam (Adaptive Moment Estimation):**
Adam is an extension of SGD that adapts the learning rates for each parameter based on their historical gradients. It combines ideas from both Momentum and RMSprop. Adam maintains two moving averages for each parameter: the first moment (mean) and the second moment (uncentered variance). These moving averages are then used to compute adaptive learning rates for each parameter.

------------------------------------------------------------------------------------------------------------------------------------------------------
# CNN Model
In this section, we will evaluate the performance of Convolutional Neural Networks (CNNs) and explore two visualization techniques: GradCAM and Saliency.

1. **CNN Performance Evaluation:**
   We will assess the effectiveness of Convolutional Neural Networks in solving our task. This involves training a CNN model on MNIST, and evaluating its accuracy on a separate test dataset.

2. **GradCAM (Gradient-weighted Class Activation Mapping):**
   GradCAM is a technique that helps visualize which parts of an image are important for the CNN's predictions. By leveraging the gradients of the target class with respect to the final convolutional layer, GradCAM highlights regions in the input image that contribute most to the prediction. We will implement GradCAM to gain insights into the CNN's decision-making process.

3. **Saliency Maps:**
   Saliency maps highlight the most relevant regions in an input image for a given prediction. We can compute the gradient of the output with respect to the input pixels to generate a saliency map. The brighter regions in the map correspond to areas that strongly influence the model's output. Saliency maps provide interpretability into the model's focus during prediction.

Overall, these techniques offer valuable insights into the CNN's internal workings and provide visualizations that help understand how the model makes decisions on specific inputs.
