import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
class Net(nn.Module):
    def __init__(self,n=1):
        super(Net, self).__init__()
        # n input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.n=n
        self.conv1 = nn.Conv2d(n, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def get_parameters(self):
        params = list(self.parameters())
        self.len=(len(params))
        self.size = params[0].size()  # conv1's .weight
        return self
    def input_32(self):
        self.input = torch.randn(1, 1, 32, 32)
        self.out = self(self.input)
        return self
    def zero_backprops(self):
        self.zero_grad()
        self.out.backward(torch.randn(1, 10))
        return self
    def loss(self):
        output = self(self.input)
        self.target = torch.randn(10)  # a dummy self.target, for example
        self.target = self.target.view(1, -1)  # make it the same shape as output
        self.criterion = nn.MSELoss() if self.n==1 else nn.CrossEntropyLoss() 
        loss = self.criterion(output, self.target)
        self.acc=(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
        self.loss=loss
        return self
    def loss_loss_momentum(self):
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    def backprop(self):
        self.zero_grad()     # zeroes the gradient buffers of all parameters
        self.g_befor_backward=self.conv1.bias.grad.tolist().copy()
        self.loss.backward()
        self.g=self.conv1.bias.grad
        return self
    def update(self):
        import torch.optim as optim
        # create your optimizer
        optimizer = optim.SGD(self.parameters(), lr=0.01,momentum=0 if self.n==1 else .9)
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        self.g_after_zero=self.g.tolist()
        output = self(self.input)
        loss = self.criterion(output, self.target)
        loss.backward()
        optimizer.step()    # Does the update
    def torch(self):
        return torch
    def interpret_size(self,size):
        # 3x32x32, i.e. 3-channel color images of 32x32 pixels in size
        l=size.split('x')
        l=map(lambda item:int(item),l)
        return {'channel': next(l), 'width': next(l), 'height': next(l)}
    def load_dict(self,PATH):
        self.load_state_dict(torch.load(PATH))
        return self
    def predict(self,images,classes):
        outputs = self(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
    def DataLoader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)
        self.trainloader=trainloader
        return self
    def perform_dataset(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100*correct / total))
        return self
    def class_difference(self,classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
    def CUDA(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)   