# Objective
This repository aims to create a CNN able to classify numbers between 0 and 999
## Table of contents
Project characteristics: [Characteristics](#characteristics) </br>
Dataset definition: [Dataset](#dataset) </br>
CNN implementation: [CNN](#cnn) </br>
Final results: [Results](results) </br>
How to use the CNN [How to](#how-to)
> [!IMPORTANT]
> To be able to follow the "How to" section, be sure you have access to a sufficiently powerful GPU in order to reduce the training time of the network. Otherwise, you can try to decrease the number of training epochs and the size of the train dataset, but this will cause performance degradation in terms of loss and accuracy.
## Characteristics
The starting dataset is derived from the classic MNIST and the CNN is strongly inspired by the well-known LeNet-5. </br>
### Dataset
During my studies I tried different approaches in order to obtain the best possible accuracy, starting from the dataset definition. </br>
To create the train dataset I chose a dynamic approach, avoiding a fixed in-memory dataset and generating each number during the actual training. 
This solution helped me to increase the variability and coverage of the dataset while reducing the memory usage.
For the test dataset, instead, I preferred a fixed in-memory dataset in order to improve reproducibility. </br></br>
Two different datasets have been implemented:
- **W dataset**: containing numbers from 0 to 999 in a single-channel image, where the two and three digit numbers were created concatenating the digits on the width dimension of the image.
- **C dataset**: created concatenating the digits on the channel dimension. </br>
  In order to maintain a consistent shape of the generated numbers (C,W,H) also one and two digit numbers must have three channels. For example, one digit numbers contains the actual digit in the first channel whereas the second and third channels are empty

### CNN
The CNN was inspired by the LeNet-5 and then fine-tuned to obtain the best possible network to classify numbers between 0 and 999. 
Below are reported all the tested CNN versions, designed to reduce the training loss and, more important, to improve the test accuracy.
> [!NOTE]
> As you can see under the description of each CNN version is reported the link to the commit containing the net, the link to the trained weights for that specific net and lastly the training loss and testing accuracy 
1. The net is a simple LeNet-5 where the number of output parameters was increased from 10 to 1000 to fit the thousand classes and the number of filters was increased in the two convolutional layers. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/c9119f9892777dd6c7e2b4ec3524ac3610a2c671) |
   [CNN dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/c9119f9892777dd6c7e2b4ec3524ac3610a2c671) |
   [Trained weight dataset W](./res/LeNet5W_1.pth) |
   [Trained weight dataset C](./res/LeNet5C_1.pth) </br>
   Results dataset W: Loss = 0.49, Accuracy = 90% </br>
   Results dataset C: Loss = 1.152, Accuracy = 73%
2. The net was modified by replacing Tanh activation functions with ReLU. The training dataset size was reduced from 500000 samples to 150000 and the epochs were increased from 2 to 3. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/23f0ffdfe4550ea66c8b05348a972947810a07d7) |
   [CNN dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/23f0ffdfe4550ea66c8b05348a972947810a07d7) |
   [Trained weight dataset W](./res/LeNet5W_2.pth) |
   [Trained weight dataset C](./res/LeNet5C_2.pth) </br>
   Results dataset W: Loss = 0.501, Accuracy = 85% </br>
   Results dataset C: Loss = 1.272, Accuracy = 65%
3. The number of convolutional layers was increased to 5. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/9c19d80f4e400ae32c9b4f2e24acca83f182c26e) |
   [CNN dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/9c19d80f4e400ae32c9b4f2e24acca83f182c26e) |
   [Trained weight dataset W](./res/LeNet5W_3.pth) |
   [Trained weight dataset C](./res/LeNet5C_3.pth) </br>
   Results dataset W: Loss = 1.088, Accuracy = 72% </br>
   Results dataset C: Loss = 4.203, Accuracy = 11%
4. Due to the poor results of the third net, the conv layers were reduced back to 3. Batch normalization and dropout were added to the conv and fully connected layers respectively.
Furthermore, the number of parameters in the fully connected layers was reduced. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/7f6708e647f426e3f338363f4e22fdd8fdcf2400) |
   [CNN dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/7f6708e647f426e3f338363f4e22fdd8fdcf2400) |
   [Trained weight dataset W](./res/LeNet5W_4.pth) |
   [Trained weight dataset C](./res/LeNet5C_4.pth) </br>
   Results dataset W: Loss = 0.385, Accuracy = 95% </br>
   Results dataset C: Loss = 1.348, Accuracy = 68%
5. Training epochs increased to 10. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/ae6febc2c9686c0274b84a7c6d51e0671d1e0845) |
   [CNN dataset C](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/ae6febc2c9686c0274b84a7c6d51e0671d1e0845) |
   [Trained weight dataset W](./res/LeNet5W_5.pth) |
   [Trained weight dataset C](./res/LeNet5C_5.pth) </br>
   Results dataset W: Loss = 0.103, Accuracy = 98% </br>
   Results dataset C: Loss = 0.219, Accuracy = 95%
6. Due to the worse results of the net on the C dataset I focused on the W dataset network replacing the SGD optimizer with ADAM. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/cca1e9f1209157b8958a7d74726499a0d0c58d3c) |
   [Trained weight dataset W](./res/LeNet5W_6.pth) </br>
   Results dataset W: Loss = 0.053, Accuracy = 98%
7. The dropout probability of the last fully connected layer was fine-tuned. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/54c75730725538a7b78e6bb627c8b6ed4d45f364) |
   [Trained weight dataset W](./res/LeNet5W_7.pth) </br>
   Results dataset W: Loss = 0.059, Accuracy = 98% (slightly better than 6th net version)
8. The training dataset was modified to generate more 3 and 2 digit numbers than 1 one digit numbers because the testing phase highlighted a higher error rate on the 2 and 3 digit numbers. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/261f561376110eb2e357364c3303ad11cc658cc5) |
   [Trained weight dataset W](./res/LeNet5W_8.pth) </br>
   Results dataset W: Loss = 0.069, Accuracy = 98%
9. As the higher error rate on 3 and 2 digit numbers persisted, I tried to push more on the generation of 2 and 3 digit numbers during training. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/0c29a18d1b6818e637d7dd845bafe5a60c6cd9b9) |
   [Trained weight dataset W](./res/LeNet5W_9.pth) </br>
   Results dataset W: Loss = 0.069, Accuracy = 98%
10. The error imbalance was not solved by the previous changes so I tried the opposite direction: increasing the probability of generating 1 digit numbers over 2 and 3 digit numbers. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/49ff93b73dc3ed6334ce1bf1b6df5d28836ce30f) |
   [Trained weight dataset W](./res/LeNet5W_10.pth) </br>
   Results dataset W: Loss = 0.076, Accuracy = 98%
11. Since error imbalance was still not solved I reverted to the best version of the net up to that point (7th). Then the 5x5 kernels were substituted with 3x3 kernels and a conv layer was added to the net trying to emulate the VGG philosophy. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/fe7caf667e5f41a96d08a2a9cd3584795033452a) |
   [Trained weight dataset W](./res/LeNet5W_11.pth) </br>
   Results dataset W: Loss = 0.064, Accuracy = 98%
12. Given the higher number of parameters in the conv layers brought by the 11th version of the net, I decided to add one fully connected layer and to increase the number of parameters of the fully connected layers as well. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/d6f29df1258f52a31f74ec3d2e9e8ad376527f17) |
   [Trained weight dataset W not available due to their size over the max github file size of 100Mb]() </br>
   Results dataset W: Loss = 0.097, Accuracy = 98%
13. 12th version of the net reverted to the 7th due to its large number of parameters and worse performance compared to the lighter version 7.
   I also added a small weight decay to the optimizer to help the net generalize better. </br>
   [CNN dataset W](https://github.com/LorenzoBacchini/DeepLearning-Project-Assignment/commit/d1b9648ca24ab8ecf83f543028527bca383c4e04) |
   [Trained weight dataset W](./res/LeNet5W_13.pth) </br>
   Results dataset W: Loss = 0.131, Accuracy = 98% </br>
### Results  
After all the above tests on the network, I decided to keep the 7th version because it is light enough to train while maintaining high accuracy (98%). </br>
Hence, the CNN implementations in cnnW and cnnC correspond to the 7th and 5th version rispectively.
# How to
Follow the steps below to train and test the CNN.
> [!NOTE]
> The instructions refers to the best network obtained during my studies, in particular the 7th version trained on W dataset.
1. Clone the repository.
2. Run test_dataset_setup to create the test dataset (you don't have to execute all the cells but only the ones related to the dataset W).
3. Open the cnnW file and execute all the cells to start the training and testing the model.
> [!WARNING]
> Pay attention to the cells after the train cell because you might not want to save and load the net weights. </br>
> If you do want to save the trained weights, you should use a different path from the ones already in use.

If you also want to test the C dataset, simply run test_dataset_setup to generate the test dataset and then run the cnnC file, following the same suggestions as above.
