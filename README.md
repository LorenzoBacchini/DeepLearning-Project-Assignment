# Objective
This repo aims to create a CNN able to classify number between 0 and 999 
## Characteristics
The starting dataset is extracted from the classic MNIST and the CNN is strongly inspired by the famous LeNet-5. </br>
### Dataset
During my studies i tried different paths in order to obtain the best possible accuracy, starting from the dataset.
To generate the train dataset i chose a dynamic approach, avoiding a fixed in memory dataset and generating each number during the actual training. 
This solution helped me to increase the variability and coverage of my dataset and to reduce the memory usage.
For the test dataset instead i preferred to have a fixed in memory dataset in order to increase reproducibility over coverage. </br></br>
Two different datasets has been implemented:
- (W dataset) containing number from 0 to 999 in a single channel image, where the two and three digit numbers were created 
concatenating the digit on the width dimension of the image.
- (C dataset) created concatenating the digits on the channel dimension. Important, in order to maintain the shape of the generated numbers always the same
also one and two digit number have three channels, for example one digit number contains the actual digit in the first channel when the second and third channels are empty
### CNN
the cnn was inspired by the LeNet-5 CNN and then fine tuned to obtain the best possible network to classify numbers between 0 and 999, 
below are reported all the CNN version tested to reduce the loss on the train dataset and the accuracy on the test dataset.
1. The net is a simple LeNet-5 where the output has been raised from 10 to 1000 to fit the thousand classes and the filters number has been increased in the two convolutional layer
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/65ed9b0dac1e311c5baf34936320f19169b06ea7)
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/704302f837fb01212c09397c1f48add911f81bc0)
   [Trained weight dataset W](./res/LeNet5W_1.pth)
   [Trained weight dataset C](./res/LeNet5W_1.pth)
2.  
# How to
Follow the subsequent step to train and test the CNN:
(the instruction refers to the best possible network obtained during my studies, in particular the seventh version of the network trained with the W dataset)
