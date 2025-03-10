import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

import matplotlib.pyplot as plt
import copy, os, time
import argparse

parser = argparse.ArgumentParser(description='COVID-19 Detection from X-ray Images')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers to train (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.00001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--dataset_path', type=str, default='./data/',
                    help='training and validation dataset')

args = parser.parse_args()

start_time = time.time()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = args.dataset_path



def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight    
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle = False,                              
                                                             sampler = sampler, num_workers=args.num_workers, pin_memory=True)   


val_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle = True,                              
                                                              num_workers=args.num_workers, pin_memory=True)   

dataloaders = {'train':train_loader, 'val':val_loader}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # 0: child, and 1: nonchild  ?? covid and non-covid??

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, batch_szie, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc = list()
    valid_acc = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_prec = 0.0
            running_rec = 0.0
            running_f1 = 0.0

            # Iterate over data.
            cur_batch_ind = 0
            for inputs, labels in dataloaders[phase]:
                # print(cur_batch_ind,"batch inputs shape:", inputs.shape)
                # print(cur_batch_ind,"batch label shape:", labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                # optimizer.zero_grad()

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

                cur_acc = torch.sum(preds == labels.data).double() / batch_szie
                cur_batch_ind += 1
                print("\npreds:", preds)
                print("label:", labels.data)
                print("%d-th epoch, %d-th batch (size=%d), %s acc= %.3f \n" % (
                epoch + 1, cur_batch_ind, len(labels), phase, cur_acc))

                if phase == 'train':
                    train_acc.append(cur_acc)
                else:
                    valid_acc.append(cur_acc)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} \n\n'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc= %.3f at Epoch: %d' % (best_acc, best_epoch + 1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, valid_acc


model_conv = torchvision.models.squeezenet1_0(pretrained=True)

num_classes = 2
# model_conv.classifier = nn.Sequential(
#     nn.Dropout(0.1),
#     nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
#     )
# # model_conv.num_classes = num_classes

model_conv.classifier = nn.Sequential (
    nn.Dropout(0.1),
    nn.Conv2d(512, num_classes, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=13, stride=1, padding= 0)
    )
model_conv.num_classes = num_classes


optimizer_conv = optim.Adam(model_conv.classifier.parameters(), lr=args.learning_rate)

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


model_conv, train_acc, valid_acc = train_model(model_conv, criterion, optimizer_conv,
                                                   exp_lr_scheduler, args.batch_size, num_epochs=args.epochs)
model_conv.eval()
torch.save(model_conv, './covid_squezeenet_epoch%d.pt' % args.epochs)

end_time = time.time()
print("total_time tranfer learning=", end_time - start_time)