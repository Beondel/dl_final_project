import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import trange
import pdb
import pickle

plt.ion()
device = 'cpu'

################        DATA        ################

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data/nature'
image_datasets = {t: datasets.ImageFolder(f'{data_dir}/{t}', data_transforms[t]) for t in ['train', 'test']}
dataloaders = {t: torch.utils.data.DataLoader(image_datasets[t], batch_size=4, shuffle=True, num_workers=4) for t in ['train', 'test']}
dataset_sizes = {t: len(image_datasets[t]) for t in ['train', 'test']}
class_names = image_datasets['train'].classes

test_image_datasets = {t: datasets.ImageFolder(f'{data_dir}/{t}', data_transforms['test']) for t in image_datasets['train'].classes}
test_dataloaders = {t: torch.utils.data.DataLoader(test_image_datasets[t], batch_size=4, shuffle=True, num_workers=4) for t in image_datasets['train'].classes}
test_dataset_sizes = {t: len(test_image_datasets[t]) for t in image_datasets['train'].classes}

arch_dir = './data/architecture'
arch_classes = next(os.walk(arch_dir))[1]

arch_image_datasets = {t: datasets.ImageFolder(f'{arch_dir}/{t}', data_transforms['test']) for t in arch_classes}
arch_dataloaders = {t: torch.utils.data.DataLoader(arch_image_datasets[t], batch_size=1, shuffle=True, num_workers=1) for t in arch_classes}
arch_dataset_sizes = {t: len(arch_image_datasets[t]) for t in arch_classes}

label_to_word = {k : v for k, v in enumerate(class_names)}


####################################################






################        Model        ################

model_ft = models.resnet18(pretrained=True)

model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#####################################################






def train(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in trange(num_epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model







################        VISUALIZATION        ################

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

################################################################






if __name__ == '__main__':
    # model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    # torch.save(model_ft.state_dict(), './model.pt')

    model_ft.load_state_dict(torch.load('./model.pt'))
    model_ft.eval()

    result_dir = './data/results/nature'

    aggregator = []

    for i, (inputs, labels) in enumerate(test_dataloaders['cloud']):
        y_hat = model_ft(inputs)
        
        guesses = torch.max(y_hat, 1)[1].numpy()
        outputs = y_hat.detach().numpy()[0]
        print(f'predicted: {[label_to_word[n] for n in guesses]}')
        print(f'outputs: {outputs}\n')

        aggregator.append(outputs.tolist())
        if i == 5:
            break
    
    A = np.array(aggregator)

    np.save(f'{result_dir}/cloud', A)
    B = np.load(f'{result_dir}/cloud.npy')
    pdb.set_trace()



    # visualize_model(model_ft)
    # plt.ioff()
    # plt.show()