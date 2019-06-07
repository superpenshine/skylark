# Neural Network Class

import pdb
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from model import ResNet18, ResNet
from pathlib import Path
from util.transform import CustomPad, GroupRandomCrop, ToTensor, Resize, LogPolartoPolar
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
        self.data_dir = str(config.h5_dir)
        self.valid_size = config.valid_size
        self.min_step_diff = config.min_step_diff
        self.max_step_diff = config.max_step_diff
        self.cuda = config.cuda
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.checkpoint_freq = config.checkpoint_freq
        self.input_size = config.input_size
        self.crop_size = config.crop_size
        self.label_size = config.label_size

        # Sizes to crop the input img
        self.ltl = (int(0.5 * (self.crop_size[0] - self.label_size[0])), int(0.5 * (self.crop_size[1] - self.label_size[1])))
        self.lbr = (self.ltl[0] + self.label_size[0], self.ltl[1] + self.label_size[1])

        self.tr_data_dir = Path(self.data_dir + "_tr.h5")
        self.va_data_dir = Path(self.data_dir + "_va.h5")

        # Global step
        self.step = 0


    def load_writer(self):
        '''
        Load the tensorboard writter
        '''
        self.writer = SummaryWriter(log_dir = str(self.log_dir), flush_secs=0)


    def load_data(self):
        '''
        Prepare train/test data
        '''
        trans = [LogPolartoPolar(),
                 Resize((self.input_size)), 
                 CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                 CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0), 
                 GroupRandomCrop(self.crop_size, label_size=self.label_size), 
                 ToTensor()
                 # transforms.ToPILImage(), 
                 # transforms.Resize((128, 128)), # Requires PIL image
                 # transforms.Pad((64, 0, 64, 0), 'constant'), # only supports RGB
                 # transforms.RandomCrop(128, 128), # won't work since we want random crop at the same posion of the three images
                 # transforms.ToTensor() # PIL to tensor
                 ]

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            transforms = trans) # RandomCrop is group op

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            transforms = trans) # RandomCrop is group op
        # np.random.seed(1234)
        # indices = list(range(len(data)))
        # np.random.shuffle(indices)
        # split = int(len(data) * (1 - self.valid_size))
        # train, valid = indices[:split], indices[split:]
        # train_sampler = SubsetRandomSampler(train)
        # valid_sampler = SubsetRandomSampler(valid)

        self.train_loader = DataLoader(self.data_tr, 
                                       batch_size = self.batch_size, 
                                       # sampler = train_sampler, 
                                       shuffle = True)
        self.valid_loader = DataLoader(self.data_va, 
                                       batch_size = self.batch_size, 
                                       # sampler = valid_sampler, 
                                       shuffle = True)


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


    def sanity_check_randcrop(self):
        '''
        train over ramdomly cropped patch from one triplets
        '''
        self.load_writer()
        self.load_model()

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op
        pad = transforms.Compose([CustomPad((8, 0, 8, 0), 'circular'), 
                                  CustomPad((0, 8, 0, 8), 'zero', constant_values=0)])
        to_tensor = ToTensor()
        g_randcroup = GroupRandomCrop(self.crop_size, label_size=self.label_size)

        i0, i1, label = self.data_tr[np.random.randint(0, len(self.data_tr)-1)]

        # test over a toy example, step at 32, 
        i0 = np.zeros((self.input_size[0], self.input_size[1], 4), dtype=np.float32)
        i0[:50] = i0[:50] + 1
        i1 = np.zeros((self.input_size[0], self.input_size[1], 4), dtype=np.float32)
        i1[:30] = i1[:30] + 1
        label_input = np.zeros((self.input_size[0], self.input_size[1], 4), dtype=np.float32)
        label_input[:40] = label_input[:40] + 1


        # i0 = cv2.resize(i0, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # i1 = cv2.resize(i1, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # label_input = cv2.resize(label, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

        i0_input = pad(i0)
        i1_input = pad(i1)
        label_input = pad(label_input)

        self.model.train()

        for iter_id in range(800):

            i0, i1, label = g_randcroup(i0_input, i1_input, label_input)
            i0 = to_tensor(i0)
            i1 = to_tensor(i1)
            label = to_tensor(label)
            duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
            i1_crop = i1[:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            i1_crop = torch.unsqueeze(i1_crop, 0)
            duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
            output = self.model(duo)
            # if output2 is not None:
            #     print(output-output2)
            #     output2 = None
            loss = self.criterion(output + i1_crop, torch.unsqueeze(label, 0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))
            # print(duo - a)
            if self.step % 5 == 0:
                print("\nvalid:")
                # print(self.model.state_dict().keys())
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
                self.model.eval()
                with torch.no_grad():
                    output2 = self.model(duo)
                    valid_loss = self.criterion(output2 + i1_crop, torch.unsqueeze(label, 0))
                    print("valid loss: {}".format(valid_loss.item()))
                self.model.train()
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Valid/Loss', valid_loss, self.step)
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
            self.step += 1
        # visualize overfit result
        var=1
        self.model.eval()

        i0_input = to_tensor(i0_input)
        i1_input = to_tensor(i1_input)
        label_input = to_tensor(label_input)
        duo = torch.unsqueeze(torch.cat([i0_input, i1_input], dim=0), 0)
        # self.ltl = (8, 8)
        # self.lbr = (self.ltl[0] + self.label_size[0], self.ltl[1] + self.label_size[1])
        i1_crop = i1_input[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        duo, i1_crop = duo.to(self.device), i1_crop.to(self.device)
        with torch.no_grad():
            output = self.model(duo)
            out = output[0] + i1_crop
        self.writer.add_image('i0', i0_input[var], dataformats='HW')
        self.writer.add_image('gt', label_input[var], dataformats='HW')
        self.writer.add_image('synthesized', out[var], dataformats='HW')
        self.writer.add_image('i1_unpadded', i1_crop[var], dataformats='HW')

        self.save()
        plt.subplot(241)
        plt.imshow(i0_input[var].to('cpu')[var])
        plt.colorboar()
        plt.title('i0')
        plt.subplot(242)
        plt.title('GT')
        plt.imshow(label_input.to('cpu')[var])   
        plt.colorbar() 
        plt.subplot(243)
        plt.title('Out')
        plt.imshow(out.to('cpu')[var])
        plt.colorbar()
        plt.subplot(244)
        plt.title('i1')
        plt.imshow(i1_crop.to('cpu')[var])
        plt.colorbar()

        plt.show()


    def sanity_check_no_randcrop(self):
        self.load_writer()
        self.load_model()

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op
        pad = transforms.Compose([CustomPad((8, 0, 8, 0), 'circular'), 
                                  CustomPad((0, 8, 0, 8), 'zero', constant_values=0)])
        random_crop = GroupRandomCrop(self.crop_size, label_size=self.label_size)
        to_tensor = ToTensor()

        i0, i1, label = self.data_tr[np.random.randint(0, len(self.data_tr)-1)]
        # Shrink to label size
        i0 = cv2.resize(i0, dsize=self.label_size, interpolation=cv2.INTER_LINEAR)
        i1 = cv2.resize(i1, dsize=self.label_size, interpolation=cv2.INTER_LINEAR)
        i1_crop = np.array(i1) # for output calculation
        label = cv2.resize(label, dsize=self.label_size, interpolation=cv2.INTER_LINEAR)
        # Pad inputs back to input size
        i0 = pad(i0)
        i1 = pad(i1)
        i0 = to_tensor(i0)
        i1 = to_tensor(i1)
        i1_crop = to_tensor(i1_crop)
        label = to_tensor(label)
        duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
        duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)

        self.model.train()

        for iter_id in range(1):
            output = self.model(duo)
            # if output2 is not None:
            #     print(output-output2)
            #     output2 = None
            loss = self.criterion(output + i1_crop, torch.unsqueeze(label, 0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))
            # print(duo - a)
            if self.step % 5 == 0:
                print("\nvalid:")
                # print(self.model.state_dict().keys())
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
                self.model.eval()
                with torch.no_grad():
                    output2 = self.model(duo)
                    valid_loss = self.criterion(output2 + i1_crop, torch.unsqueeze(label, 0))
                    print("valid loss: {}".format(valid_loss.item()))
                self.model.train()
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Valid/Loss', valid_loss, self.step)
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
            self.step += 1
        # visualize overfit result
        var=1
        self.model.eval()
        i1_crop = i1_crop
        with torch.no_grad():
            output = self.model(duo)
            out = output[0] + i1_crop
        # pdb.set_trace()
        self.writer.add_image('i0', i0[var], dataformats='HW')
        self.writer.add_image('gt', label[var], dataformats='HW')
        self.writer.add_image('synthesized', out[var], dataformats='HW')
        self.writer.add_image('i1_unpadded', i1_crop[var], dataformats='HW')

        self.save()

    def single_batch_train(self):
        '''
        Overfit over one batch, for loss check only
        '''
        output2 = None

        print("\ntrain:")
        self.model.train()
        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            input1 = i0
            i1_crop = i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            duo = torch.cat([i0, i1], dim=1)
            duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
            # a = torch.tensor(duo)
            break
        for iter_id in range(400):
            output = self.model(duo)
            # if output2 is not None:
            #     print(output-output2)
            #     output2 = None
            loss = self.criterion(output + i1_crop, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))
            # print(duo - a)
            if self.step % 5 == 0:
                print("\nvalid:")
                # print(self.model.state_dict().keys())
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
                self.model.eval()
                with torch.no_grad():
                    output2 = self.model(duo)
                    valid_loss = self.criterion(output2 + i1_crop, label)
                    print("batch{}, loss: {}".format(b_id, valid_loss.item()))
                self.model.train()
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Valid/Loss', valid_loss, self.step)
                # print(self.model.state_dict()['layer1.0.bn.running_mean'])
            self.step += 1
        self.save()
        # visualize overfit result
        var=1
        self.model.eval()
        i1_crop = i1_crop[0]
        with torch.no_grad():
            output = self.model(duo)
            out = output[0] + i1_crop
        # pdb.set_trace()
        self.writer.add_image('step2.1/i0_randcrop', input1[0,var], dataformats='HW')
        self.writer.add_image('step2.2/gt_randcrop', label[0,var], dataformats='HW')
        self.writer.add_image('step4/synthetic', out[var], dataformats='HW')
        self.writer.add_image('step5/i1_centercrop', i1_crop[var], dataformats='HW')


    def train(self):
        '''
        Train the model
        epoch_loss: total train loss
        '''
        print("\ntrain:")
        self.model.train()

        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            # # Convert to tensor
            # i0, i1, label, i1_crop = torch.from_numpy()
            # Only cut i1 for err calc
            i1_crop = i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            # Concatenate two input imgs in NCHW format
            duo = torch.cat([i0, i1], dim=1)
            duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
            output = self.model(duo)
            # print(self.model.state_dict().keys())
            # pdb.set_trace()
            # Avg per img loss: Err = f(I0,I1) + I1 - I0.5
            loss = self.criterion(output + i1_crop, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))

            if self.step % 10 == 0:
                valid_result = self.valid()
                self.model.train()
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Valid/Loss', valid_result, self.step)
            self.step += 1


    def valid(self):
        '''
        Test the accuracy of the current model parameters
        '''
        print("\nvalid:")
        self.model.eval()
        valid_loss = 0
        num_batch = 0

        with torch.no_grad():
            for b_id, (i0, i1, label) in enumerate(self.valid_loader):
                # Only cut i1 for err calc
                i1_crop = i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
                # Concatenate two input imgs in NCHW format
                duo = torch.cat([i0, i1], dim=1)
                duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
                output = self.model(duo)
                # Err = f(I0,I1) + I1 - I0.5
                # L1 Loss
                loss = self.criterion(output + i1_crop, label)
                valid_loss += loss.item()
                print("batch{}, loss: {}".format(b_id, loss.item()))
                num_batch = b_id + 1
                if b_id == 5:
                    break

        valid_loss /= num_batch

        return valid_loss


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


    def load(self, **kwargs):
        '''
        Load pretrained model
        '''
        self.model = torch.load(self.model_dir, **kwargs)
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

        sample_input=(torch.rand(1, 8, self.crop_size[0], self.crop_size[1]))
        self.writer.add_graph(model = ResNet(), input_to_model=sample_input)

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
            # train_result = self.train()
            train_result = self.train()
            print("Epoch {} loss: {}".format(epoch, train_result))
            # accuracy = max(accuracy, valid_result[1])
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


    def test_single(self):
        '''
        Test the model using single inputj, for illustration
        '''
        var = 1
        n_row = 5

        self.load_writer()

        pad = transforms.Compose([CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                                  CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0)])
        to_tensor = ToTensor()
        data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            verbose = True) # RandomCrop is group op

        data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            verbose = True) # RandomCrop is group op

        # Normalized triplet without transform
        i0, i1, label = data_tr[np.random.randint(0, len(data_tr)-1)]
        i0 = cv2.resize(i0, dsize=self.input_size, interpolation=cv2.INTER_LINEAR)
        i1 = cv2.resize(i1, dsize=self.input_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=self.input_size, interpolation=cv2.INTER_LINEAR)
        i1_label_sized = i1
        self.writer.add_images('triplet', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')

        i0 = pad(i0)
        i1 = pad(i1)
        i0 = to_tensor(i0)
        i1 = to_tensor(i1)
        label = to_tensor(label)
        i1_label_sized = to_tensor(i1_label_sized)

        device = torch.device('cpu')
        if Path(self.model_dir).exists():
            self.load(map_location=device)
        else:
            print("Model file does not exists, trying checkpoint")
            self.model = ResNet().to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
            self.criterion = L1Loss().to(device)
            self.load_checkpoint()

        self.model.eval()
        with torch.no_grad():
            duo = torch.cat([i0, i1], dim=0)
            duo = torch.unsqueeze(duo, 0)
            output = self.model(duo)
            out = output[0] + i1_label_sized
            residue = out - label
            original_diff = i1_label_sized - label

        out = out[var]
        label = label[var]
        residue = residue[var]
        original_diff = original_diff[var]
        i1_label_sized = i1_label_sized[var]
        # pdb.set_trace()
        self.writer.add_image('residue', (residue + torch.min(residue)) / (torch.max(residue) - torch.min(residue)), dataformats='HW')
        self.writer.add_image('synthetic', out, dataformats='HW')
        self.writer.add_image('i1_centercrop', i1_label_sized, dataformats='HW')

        # default dpi 6.4, 4.8
        # plt.figure(figsize=(20, 4), dpi=200)
        plt.subplot(241)
        plt.title('GT')
        plt.imshow(label)   
        plt.colorbar() 
        plt.subplot(242)
        plt.title('Out')
        plt.imshow(out)
        plt.colorbar()
        plt.subplot(243)
        plt.title('i1_cropped')
        plt.imshow(i1_label_sized)
        plt.colorbar()

        vmax = max(torch.max(residue), torch.max(original_diff))
        vmin = min(torch.min(residue), torch.min(original_diff))
        plt.subplot(244)
        plt.title('Out-GT')
        plt.imshow(residue, vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.subplot(248)
        plt.title('i1_cropped-GT')
        plt.imshow(original_diff, vmin = vmin, vmax = vmax)
        plt.colorbar()

        plt.show()
        # plt.savefig("a.png")
        print("Residue sum: ", torch.sum(torch.abs(residue)))
        print("original diff: ", torch.sum(torch.abs(original_diff)))
