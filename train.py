import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from absl import app

# data importer
class CIFAR10(Dataset):
    base_folder = "cifar-10-batches-py"

    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    test_list = [
        "test_batch"
    ]

    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(self, root, classes = 'all', train=True, transform=None):
        self.transform = transform
        self.train = train
        self.root = root
        self.classes = classes
        self.data, self.labels = self._load_batch(classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_batch(self, classes):
        self._load_meta()

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        data_list = []
        labels_list = []

        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                cifar_dict = pickle.load(f, encoding='latin1')
            data = cifar_dict['data']
            labels = np.array(cifar_dict['labels'])

            if classes != 'all':
            # Filter samples from specified classes
                mask = np.isin(labels, list(self.class_to_idx.values()))
                data_list.append(data[mask])
                labels = labels[mask]
                labels = [self.idx_to_enc.get(x, x) for x in labels]
                labels_list.append(labels)
            else:
                data_list.append(data)
                labels_list.append(labels)


        data = np.concatenate(data_list)
        labels = np.concatenate(labels_list)
        data = data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)) 
        return data, labels
    
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            classes = data[self.meta["key"]]
        
        if self.classes != 'all':
            self.class_to_idx = {}
            self.idx_to_enc = {}
            for i, item in enumerate(self.classes):
                if item in classes:
                    self.class_to_idx.update({item: classes.index(item)})
                    self.idx_to_enc.update({classes.index(item):i})
        else:
            self.classes = classes
            self.class_to_idx = {_class: i for i, _class in enumerate(classes)}

# defining the network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 1)
        self.conv2 = nn.Conv2d(6, 16, 1)
        self.conv3 = nn.Conv2d(16, 24, 1)
        self.conv4 = nn.Conv2d(24, 32, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# converting tensor to image
def tensor_to_img(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    
    return np.transpose(npimg, (1, 2, 0))
    
def main(_):
    params = {
       "epochs": 30, 
       "batch_size": 5,
       "lr": 1e-4,
       "noise": 3e-4, 
       "file": '../../Downloads',
       "classes": ['deer', 'truck'],
        "valid_ratio": 0.2
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations to be applied to the dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),    # Convert numpy array to PIL Image
        transforms.ToTensor(),      # Convert PIL Image to tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the data
    ])

    train_ds = CIFAR10(params['file'], classes=params['classes'], transform=transform)
    test_ds = CIFAR10(params['file'], classes=params['classes'], train=False, transform=transform)

    # Creating data indices for training and validation splits:
    dataset_size = len(train_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(params['valid_ratio'] * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_ds, 
                            batch_size=params['batch_size'], 
                            sampler=train_sampler, 
                            num_workers=8)

    valid_loader = DataLoader(train_ds, 
                            batch_size=params['batch_size'], 
                            sampler=valid_sampler, 
                            num_workers=8)

    test_loader = DataLoader(test_ds, 
                            batch_size=params['batch_size'], 
                            shuffle=False, 
                            num_workers=8)

    net = Net().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])

    loss_train = np.zeros(params['epochs'])
    acc_train = np.zeros(params['epochs'])

    loss_valid = np.zeros(params['epochs'])
    acc_valid = np.zeros(params['epochs'])

    # training loop
    for epoch in range(params['epochs']): 
        net.train()
        total_samples = 0
        running_loss = 0.0
        correct_predictions = 0
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item()

        # validation step after each epoch
        net.eval()
        valid_samples = 0
        valid_loss = 0.0
        valid_predictions = 0 
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(inputs)

                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                valid_predictions += (predicted == labels).sum().item()
                valid_samples += labels.size(0)

                valid_loss += loss.item()
                
        loss_train[epoch] = running_loss / len(train_loader)
        acc_train[epoch] = correct_predictions / total_samples

        loss_valid[epoch] = valid_loss / len(valid_loader)
        acc_valid[epoch] = valid_predictions / valid_samples

        print(f'[Epoch {epoch + 1:02d}] train > loss: {loss_train[epoch]:.5f}, accuracy: {acc_train[epoch]:.3f}; valid > loss: {loss_valid[epoch]:.5f}, accuracy: {acc_valid[epoch]:.3f}')

    # ploting loss and accuracy for training and validation
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(121)
    ax0.plot(list(range(params['epochs'])), loss_train, label = 'train')
    ax0.plot(list(range(params['epochs'])), loss_valid, label = 'train')

    ax0.set_xlabel('epochs')
    ax0.set_ylabel('loss')
    ax0.legend()

    ax1 = fig0.add_subplot(122)
    ax1.plot(list(range(params['epochs'])), acc_train, label = 'train')
    ax1.plot(list(range(params['epochs'])), acc_valid, label = 'train')

    ax1.set_xlabel('accuracy')
    ax1.set_ylabel('loss')
    ax1.legend()

    # testing 
    test_predictions = 0
    test_samples = 0
    test_loss = 0.0
    pred_list = []
    label_list = []

    net.eval()
    for data in test_loader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        label_list.append(data[1])
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.detach().to('cpu').numpy())
        test_predictions += (predicted == labels).sum().item()
        test_samples += labels.size(0)

        test_loss += loss.item()

    loss_test = running_loss / len(train_loader)
    acc_test = correct_predictions / total_samples

    print(f'test > loss: {loss_test:.5f}, accuracy: {acc_test:.3f}')

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    # final list contains 0: correct deer, 1: correct truck, 2: wrong deer, 3 wrong truck
    final_list = np.abs(3*label_list - 2*pred_list)

    imgs = []

    # find randomly: 2 correct deers and trucks, 2 wrong deers and trucks in order 
    for value in range(4):
        ind = np.where(final_list == value)[0]
        ind = np.random.choice(ind, 2, replace=False)

        for x in ind:
            imgs.append(tensor_to_img(test_ds[x][0]))

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 4)

    # Loop through the images and plot them in subplots
    for i, image in enumerate(imgs):
        ax = axs[i // 4, i % 4]
        ax.imshow(image)  # Replace 'image' with your image data
        ax.axis('off')    # Turn off axis labels and ticks

        if i in [0,1,2,3]:
            ax.set_title('correct predict')
        else:
            ax.set_title('wrong predict')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
  app.run(main)

