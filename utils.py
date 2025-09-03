from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from matplotlib.lines import Line2D

from utils_datasets import mnist_mean
from utils_datasets import mnist_std

to_pil = transforms.ToPILImage()

# Function that take in input the image path, a trained network, and a transform function
# and uses the given net to predict the image class
def predict_img(img_path, net, transform):
    # image loading part
    img = Image.open(img_path).convert("L")
    plt.imshow(img)
    img = transform(img)
    img = img.unsqueeze(0)

    # img evaluation part
    net.eval()
    with torch.no_grad():
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        print(f'the predicted class is: {predicted[0]}')

# function to get the number of correct prediction given the predictions and labels tensors 
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# given a trained network and a set of images return the predictions 
# and the probability of belonging to the predicted class
def images_to_probs(net, images):
    
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

# function that return a figure depicting 4 test images related with the predictions
# of the net and the probability of the obtained predictions
# the right predictions are highlighted in green whereas the wrong ones in red
def plot_classes_preds(net, images, labels, classes, denormalize):
    preds, probs = images_to_probs(net, images)
    # plot the first 4 images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(28, 28))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        # plotting the image
        matplotlib_imshow(images[idx], denormalize)
        # plotting the informations (prediction, prediction probability, label)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# function to plot a given image
def matplotlib_imshow(img, denormalize):
    img = img.cpu()
    img = denormalize(img, mnist_std, mnist_mean)
    
    plt.imshow(to_pil(img))

# function that computes pr (precision-recall) curve of a given class
# given the actual labels and the test probabilities
def add_pr_curve_tensorboard(class_index, test_probs, test_label, classes, writer, global_step=0):
    # extracting the test_label indices corresponding to the current class (class_index)
    tensorboard_truth = test_label == class_index
    # extracting the predicted probabilities for the current class (class_index)
    tensorboard_probs = test_probs[:, class_index]

    # computing the precision-recall curve
    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# get gradients for each parameter in the network and plot their histograms
def add_gradient_hist(net):
    ave_grads = [] 
    layers = []
    # for each parameter in the network excluding biases
    for n,p in net.named_parameters():
        if ("bias" not in n):
            layers.append(n)
            if p.requires_grad: 
                # get the average gradient for the parameter
                ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
            else:
                ave_grad = 0
            ave_grads.append(ave_grad)
        
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]

    # plotting the average gradients
    fig = plt.figure(figsize=(12, 12))
    plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.legend([Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    
    return fig