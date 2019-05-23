# Neural Network Class

import os
import numpy as np
from model import ResNet18, ResNet
from pathlib import Path
from util.transform import CustomPad, GroupRandomCrop
from dataset.Astrodata import Astrodata

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import L1Loss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler


class network(object):
    """docstring for network"""
    def __init__(self, config):
        super(network, self).__init__()
        self.checkpoint = "checkpoint.tar"
        self.model_dir = "model.pth"
        self.log_dir = config.log_dir
        self.data_dir = config.h5_dir
        self.valid_size = config.valid_size
        self.min_step_diff = config.min_step_diff
        self.max_step_diff = config.max_step_diff
        self.cuda = config.cuda
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.checkpoint_freq = config.checkpoint_freq
        self.input_size = config.input_size
        self.label_size = config.label_size

        # Sizes to crop the input img
        self.ltl = (int(0.5 * (self.input_size[0] - self.label_size[0])), int(0.5 * (self.input_size[1] - self.label_size[1])))
        self.lbr = (self.ltl[0] + self.label_size[0], self.ltl[1] + self.label_size[1])

        # Global step
        self.step = 0


    def load_writer(self):
        '''
        Load the tensorboard writter
        '''
        self.writer = SummaryWriter(log_dir = str(self.log_dir))


    def load_data(self):
        '''
        Prepare train/test data
        '''
        trans = [CustomPad((0, int((self.input_size[1]-1)*0.5), 0, int((self.input_size[1]-1)*0.5)), 'circular'), 
                 CustomPad((int((self.input_size[0]-1)*0.5), 0, int((self.input_size[0]-1)*0.5), 0), 'zero', constant_values=0), 
                 GroupRandomCrop(self.input_size, label_size=self.label_size), 
                 transforms.ToPILImage(), 
                 # transforms.Resize((128, 128)), # Using RandomCrop now
                 # transforms.Pad((64, 0, 64, 0), 'constant'), # only supports RGB
                 # transforms.RandomCrop(128, 128), # won't work since we want random crop at the same posion of the three images
                 transforms.ToTensor()]

        data = Astrodata(self.data_dir, 
                         min_step_diff = self.min_step_diff, 
                         max_step_diff = self.max_step_diff, 
                         rtn_log_grid = False, 
                         transforms = trans, 
                         group_trans_id = [2]) # RandomCrop is group op

        np.random.seed(1234)
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        split = int(len(data) * self.valid_size)
        train, valid = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train)
        valid_sampler = SubsetRandomSampler(valid)

        self.train_loader = DataLoader(data, 
                                       batch_size = self.batch_size, 
                                       sampler = train_sampler)
        self.valid_loader = DataLoader(data, 
                                       batch_size = self.batch_size, 
                                       sampler = valid_sampler)


    def load_model(self):
        '''
        Load the torch model
        '''
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        sample_input=(torch.rand(1, 8, self.input_size[0], self.input_size[1])) # To trace the flow sizes
        self.writer.add_graph(model = ResNet(), input_to_model=sample_input)
        self.model = ResNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = L1Loss().to(self.device)


    def train(self):
        '''
        Train the model
        train_loss: total train loss
        total: total number of inputs trained
        '''
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            # Only cut i1 for err calc
            i1_crop = i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            # Concatenate two input imgs in NCHW format
            duo = torch.cat([i0, i1], dim=1)
            duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(duo)

            # Avg per img loss: Err = f(I0,I1) + I1 - I0.5
            # import pdb
            # pdb.set_trace()
            loss = self.criterion(output + i1_crop, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += self.batch_size
            print(total, loss.item())
            if self.step % 5 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
            self.step += 1

        return train_loss


    def load_checkpoint(self):
        '''
        Load the model from checkpoing
        '''
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy = checkpoint['accuracy']
        self.step = checkpoint['step']

        return accuracy, epoch


    def save_checkpoint(self, accuracy, epoch):
        ''' 
        Create a checkpoint with additional info: current accuracy and current epoch
        '''
        torch.save({
            'epoch': epoch,
            'step': self.step,
            'accuracy': accuracy, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.checkpoint)
        print("Checkpoint saved")


    def load(self):
        '''
        Load pretrained model
        '''
        torch.load(self.model_dir)
        print("Model loaded from  {}".format(self.model_dir))


    def save(self):
        '''
        Save the model when all epochs trained
        '''
        torch.save(self.model, self.model_dir)
        print("Model saved to {}".format(self.model_dir))

        
    def run(self):
        '''
        Function to drive the training process
        Calling this Object.run() to start
        '''
        self.load_writer()
        self.load_data()
        self.load_model()
        accuracy = 0
        start_epoch = 1

        # Check if previous checkpoint exists. If it does, load the checkpoint
        if Path(self.checkpoint).exists():
            accuracy, epoch = self.load_checkpoint()
            start_epoch = epoch + 1
        # Iterate through epochs
        for epoch in range(start_epoch, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: {}/{}".format(epoch, self.epochs))
            train_result = self.train()
            print("Epoch {} loss: {}".format(epoch, train_result))
            # test_result = self.test()
            # accuracy = max(accuracy, test_result[1])
            # Save checkpoint periodically
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(accuracy, epoch)

        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
        self.save()
        # Remove the checkpoint when training finished and the model is saved
        print("Checkpoint removed upon training complete")
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
