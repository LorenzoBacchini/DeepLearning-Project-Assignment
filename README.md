# Objective
This repo aims to create a CNN able to classify number between 0 and 999
## Table of contents
Project characteristics: [Characteristics](#characteristics) </br>
Dataset definition: [Dataset](#dataset) </br>
CNN implementation: [CNN](#cnn) </br>
Final results: [Results](results) </br>
How to use the cnn on your own [How to](#how-to)
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
1. The net is a simple LeNet-5 where the output was raised from 10 to 1000 to fit the thousand classes and the filters number were increased in the two convolutional layer </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/c9119f9892777dd6c7e2b4ec3524ac3610a2c671) |
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/c9119f9892777dd6c7e2b4ec3524ac3610a2c671) |
   [Trained weight dataset W](./res/LeNet5W_1.pth) |
   [Trained weight dataset C](./res/LeNet5C_1.pth) </br>
   Results dataset W: Loss = 0.49, Accuracy = 90% </br>
   Results dataset C: Loss = 1.152, Accuracy = 73%
2. The net was modified substituting Tanh activation functions with ReLU, the training set was reduced from 500000 samples to 150000 and the epoch raised from 2 to 3 </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/23f0ffdfe4550ea66c8b05348a972947810a07d7) |
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/23f0ffdfe4550ea66c8b05348a972947810a07d7) |
   [Trained weight dataset W](./res/LeNet5W_2.pth) |
   [Trained weight dataset C](./res/LeNet5C_2.pth) </br>
   Results dataset W: Loss = 0.501, Accuracy = 85% </br>
   Results dataset C: Loss = 1.272, Accuracy = 65%
3. The net convolutional layers was raised to 5 </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/9c19d80f4e400ae32c9b4f2e24acca83f182c26e) |
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/9c19d80f4e400ae32c9b4f2e24acca83f182c26e) |
   [Trained weight dataset W](./res/LeNet5W_3.pth) |
   [Trained weight dataset C](./res/LeNet5C_3.pth) </br>
   Results dataset W: Loss = 1.088, Accuracy = 72% </br>
   Results dataset C: Loss = 4.203, Accuracy = 11%
4. Due to the poor results of the third net the conv layers were brought back to 3 but batch normalization and dropout were added to the conv and fully connected layers respectively.
Furthermore the fully connected layers parameters was reduced </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/7f6708e647f426e3f338363f4e22fdd8fdcf2400) |
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/7f6708e647f426e3f338363f4e22fdd8fdcf2400) |
   [Trained weight dataset W](./res/LeNet5W_4.pth) |
   [Trained weight dataset C](./res/LeNet5C_4.pth) </br>
   Results dataset W: Loss = 0.385, Accuracy = 95% </br>
   Results dataset C: Loss = 1.348, Accuracy = 68%
5. Training epochs raised to 10 </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/ae6febc2c9686c0274b84a7c6d51e0671d1e0845) |
   [File dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/ae6febc2c9686c0274b84a7c6d51e0671d1e0845) |
   [Trained weight dataset W](./res/LeNet5W_5.pth) |
   [Trained weight dataset C](./res/LeNet5C_5.pth) </br>
   Results dataset W: Loss = 0.103, Accuracy = 98% </br>
   Results dataset C: Loss = 0.219, Accuracy = 95%
6. Due to the worse results of the net on the C dataset I focused more on the W dataset network substituting the SGD optimizer with ADAM </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/cca1e9f1209157b8958a7d74726499a0d0c58d3c) |
   [Trained weight dataset W](./res/LeNet5W_6.pth) </br>
   Results dataset W: Loss = 0.053, Accuracy = 98%
7. The dropout probability of the last fully connected layer was fine tuned </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/54c75730725538a7b78e6bb627c8b6ed4d45f364) |
   [Trained weight dataset W](./res/LeNet5W_7.pth) </br>
   Results dataset W: Loss = 0.059, Accuracy = 98%
8. The training dataset was modified to generate more 3 and 2 digit numbers than 1 one digit numbers because the testing phase highlighted an higher error on the 2 and 3 digit numbers </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/261f561376110eb2e357364c3303ad11cc658cc5) |
   [Trained weight dataset W](./res/LeNet5W_8.pth) </br>
   Results dataset W: Loss = 0.069, Accuracy = 98%
9. Since the higher error on 3 and 2 digit numbers was still present i tried to push more on the generation of 2 and 3 digit numbers during training </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/0c29a18d1b6818e637d7dd845bafe5a60c6cd9b9) |
   [Trained weight dataset W](./res/LeNet5W_9.pth) </br>
   Results dataset W: Loss = 0.069, Accuracy = 98%
10. The error unbalance was not solved by the previous changes so i tried to move in the opposite direction increasing the probability of generating 1 digit numbers over 2 and 3 digit numbers </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/49ff93b73dc3ed6334ce1bf1b6df5d28836ce30f) |
   [Trained weight dataset W](./res/LeNet5W_10.pth) </br>
   Results dataset W: Loss = 0.076, Accuracy = 98%
11. Since error unbalance was still present i decided to revert to the best version of the net up to that point (7th version) and substitute the 5x5 kernels with 3x3 kernels. I also added a conv layer to the net </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/fe7caf667e5f41a96d08a2a9cd3584795033452a) |
   [Trained weight dataset W](./res/LeNet5W_11.pth) </br>
   Results dataset W: Loss = 0.064, Accuracy = 98%
12. Added one fully connected layer and increased the number of parameters of the fully connected layers </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/d6f29df1258f52a31f74ec3d2e9e8ad376527f17) |
   [Trained weight dataset W not available due to their size over the max github file size of 100Mb]() </br>
   Results dataset W: Loss = 0.097, Accuracy = 98%
13. 12th version of the net reverted to the 7th version due to its large size in term of parameters and its bad performance over the lighter 7th version of it.
   I also added a small weight decay on the optimizer to help the net to generalize better  </br>
   [File dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/d1b9648ca24ab8ecf83f543028527bca383c4e04) |
   [Trained weight dataset W](./res/LeNet5W_13.pth) </br>
   Results dataset W: Loss = 0.131, Accuracy = 98% </br>
### Results  
After all the above test on the network I decided to keep the seventh version of the net because it's light enough to train but also accurate (98% accuracy). </br> 
> [!IMPORTANT]
> The net showed in the cnnW and cnnC files is therefore the 7th version
# How to
Follow the subsequent step to train and test the CNN:
> [!NOTE]
> The instruction refers to the best possible network obtained during my studies, in particular the seventh version of the network trained with the dataset W
1. clone the repo
2. run the test_dataset_setup to create the test dataset (you don't have to execute all the cells but only the one referred to the dataset W)
3. open the cnnW file and execute all the cells to start the training and test the model
> [!WARNING]
> Pay attention to the cells after the train cell because you might not want to save and load the net weight and if you want to save the trained weights you should use a different path to save the weights than the ones already in use

If you want to test also the dataset C simply run the test_dataset_setup to generate the test dataset and then run the cnnC file with the same suggestions of above
