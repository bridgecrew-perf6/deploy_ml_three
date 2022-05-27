import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, 
         test_loader,
         criterion,
         device):
    """
    Function for testing model performance
    """
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    
    logger.info(f"Test loss: {total_loss}")
    logger.info(f"Test accuracy: {total_acc}")

def train(model, 
          train_loader,
          validation_loader,
          criterion, 
          optimizer,
          epochs,
          device):
  
    best_loss=1e6
    image_dataset={'train':train_loader, 
                   'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(1, epochs +1 ):
        program_starts = time.time()
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

            now = time.time()
            logger.info(f'{phase} loss: {epoch_loss}, acc: {epoch_acc}, best loss: {best_loss}')
            logger.info(f"epoch time: {now - program_starts}")

        if loss_counter==1:
            break
#         if epoch==0:
#             break
    return model
    
def net():
    model = models.resnet50(pretrained=True) #Using a Resnet 50 Pre-Trained Model
    num_features = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(num_features, 256),
                   nn.ReLU(inplace=True),
                   nn.Linear(256, 133))
    return model


def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader



def main(args):
    logger.info(f'Input args: {args}')
    
    # see if GPU is available 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on: {device}")

    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    
    model=net()
    model=model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.AdamW(model.fc.parameters(), 
                            lr=args.lr, 
                            eps= args.eps, 
                            weight_decay = args.weight_decay)
    
    logger.info("Starting Model Training")
    model=train(model, 
                train_loader,
                validation_loader,
                criterion,
                optimizer,
                args.epochs,
               device)
    
    logger.info("Testing Model")
    test(model, 
         test_loader, 
         criterion,
        device)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size" )
    parser.add_argument( "--epochs", type=int, default=2, metavar="N", help="number of epochs to train")
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate" )
    parser.add_argument( "--eps", type=float, default=1e-8, metavar="EPS", help="eps" )
    parser.add_argument( "--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coef" )
    
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
   
    args=parser.parse_args()
    main(args)
