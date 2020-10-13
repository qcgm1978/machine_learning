import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from utilities import saveAndShow
class Classifier(object):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    def __init__(self, net):
        self.net = net
    
        # functions to show an image
    def imshow(self,img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        saveAndShow(plt)
        return self
    def show_random_imgs(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()
        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
    def feed_inputs(self,is_debug=False):
        import torch.optim as optim
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == (19 if is_debug else 1999):    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    # only execute 2000 to test
                    if is_debug:
                        return self
        print('Finished Training')
        return self
    def save(self):
        PATH =self.get_path()
        torch.save(self.net.state_dict(), PATH)
        return self

    def get_path(self):
        return './test/pytorch/cifar_net.pth'
    def display_test_img(self):
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        # print images
        self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
        self.images=images
        return self
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
