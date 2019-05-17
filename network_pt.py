# Neural Network Class

import numpy as np
from model import ResNet18
from pathlib import Path
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
        self.data_dir = config.h5_dir
        self.valid_size = config.valid_size
        self.min_step_diff = config.min_step_diff
        self.max_step_diff = config.max_step_diff
        self.cuda = config.cuda
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.checkpoint_freq = config.checkpoint_freq


    def load_data(self):
        '''
        Prepare train/test data
        '''
        # ex: outputsize = 128 * 128
        #     inputsize  = 130 * 130
        #     pad 150 pixels
        trans = transforms.Compose([transforms.ToPILImage(), 
                                    # transforms.Resize((128, 128)),
                                    transforms.Pad((0, 150, 0, 150), 'circular'), 
                                    transforms.Pad((150, 0, 150, 0), 'constant'), 
                                    transforms.RandomCrop(130, 130), 
                                    transforms.ToTensor()])

        data = Astrodata(self.data_dir, 
                         min_step_diff = self.min_step_diff, 
                         max_step_diff = self.max_step_diff, 
                         rtn_log_grid = False, 
                         transform = trans)

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
            # Concatenate two input imgs in NCHW format
            duo = torch.cat([i0, i1], dim=1)
            duo, label = duo.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(duo)
            import pdb
            pdb.set_trace()
            # Avg per img loss
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

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

        return accuracy, epoch


    def save_checkpoint(self, accuracy, epoch):
        ''' 
        Create a checkpoint with additional info: current accuracy and current epoch
        '''
        torch.save({
            'epoch': epoch,
            'accuracy': accuracy, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.checkpoint)
        print("Checkpoint saved")

        
    def run(self):
        '''
        Function to drive the training process
        Calling this Object.run() to start
        '''
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
            print(train_result)
            # test_result = self.test()
            # accuracy = max(accuracy, test_result[1])
            # Save checkpoint periodically
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(accuracy, epoch)
            break
        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
        # Remove the checkpoint when training finished and the model is saved
        print("Checkpoint removed upon training complete")
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
