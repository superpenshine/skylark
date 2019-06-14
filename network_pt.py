# Neural Network Class

import pdb
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from model import ResNet18, ResNet
from pathlib import Path
from util.transform import CustomPad, GroupRandomCrop, ToTensor, Resize, LogPolartoPolar, Normalize, Crop
from dataset.Astrodata import Astrodata

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler


class network(object):
    """docstring for network"""
    def __init__(self, config):
        '''
        ltl: label top left pos on i1
        lbr: label bot right pos on i1
        '''
        super(network, self).__init__()
        self.checkpoint = "checkpoint.tar"
        self.model_dir = "model.pth"
        self.data_dir = str(config.h5_dir_linux)
        self.log_dir = config.log_dir
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
        # For debug on Windows
        if os.name == 'nt':
            self.data_dir = str(config.h5_dir_win)
            self.batch_size = 2
            self.valid_required = False
            self.epochs = 1
            self.min_step_diff = None

        self.tr_data_dir = Path(self.data_dir + "_tr.h5")
        self.va_data_dir = Path(self.data_dir + "_va.h5")
        # Sizes to crop the input img
        self.ltl = (int(0.5 * (self.crop_size[0] - self.label_size[0])), int(0.5 * (self.crop_size[1] - self.label_size[1])))
        self.lbr = (self.ltl[0] + self.label_size[0], self.ltl[1] + self.label_size[1])
        self.valid_required = True
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
        trans = [
                 # # For Cropped input
                 # Crop((0, 0), (440, 1024)), # should be 439x1024
                 # Resize((55, 128)),
                 # # LogPolartoPolar(), # Use polar data instead, too expensive
                 # CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                 # CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0), 
                 # GroupRandomCrop(self.crop_size, label_size=self.label_size), 
                 # Normalize(mean=.5),
                 # ToTensor()

                 # For square input
                 Resize((self.input_size)), 
                 # LogPolartoPolar(), # Use polar data instead, too expensive
                 CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                 CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0), 
                 GroupRandomCrop(self.crop_size, label_size=self.label_size), 
                 Normalize(mean=.5),
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

        # Randomly shuffle and split data to train/valid
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
        self.criterion = MSELoss().to(self.device) # set reduction=sum, or too smal to see
        # self.criterion = L1Loss().to(self.device)


    def sanity_visualize(self):
        var=1

        self.load_writer()
        self.load_model()
        device = torch.device('cpu')
        if Path(self.model_dir).exists():
            self.load(map_location=device)
        else:
            raise RuntimeError("Model.pth not found")

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)
        tran = transforms.Compose([
                                  Resize(self.input_size), 
                                  Normalize(mean=.5), 
                                  CustomPad((8, 0, 8, 0), 'circular'), 
                                  CustomPad((0, 8, 0, 8), 'zero', constant_values=0), 
                                  ])
        to_tensor = ToTensor()
        # g_randcroup = GroupRandomCrop(self.crop_size, label_size=self.label_size)
        i0, i1, label = self.data_tr[20]
        i0_in = tran(i0)
        i1_in = tran(i1)
        label_in = tran(label)
        i0 = to_tensor(i0_in)
        i1 = to_tensor(i1_in)
        label = to_tensor(label_in)

        self.model.eval()
        duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
        i0_crop = i0[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        i1_crop = i1[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        duo_crop = torch.unsqueeze(torch.cat([i0_crop, i1_crop], dim=0), 0)

        with torch.no_grad():
            output = self.model(duo)
            out = output[0] + i1_crop
            # pdb.set_trace()
            residue = out - label[:,8:-8,8:-8]

        norm = Normalize()
        self.writer.add_image('i0', norm(i0[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('gt', norm(label[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('synthesized', norm(out)[var], dataformats='HW')
        self.writer.add_image('i1_cropped', norm(i1_crop)[var], dataformats='HW')
        plt.subplot(241)
        plt.imshow(i0_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.title('i0')
        plt.subplot(242)
        plt.title('GT')
        plt.imshow(label_in[8:-8,8:-8,var])   
        plt.colorbar() 
        plt.subplot(243)
        plt.title('Out')
        plt.imshow(out[var])
        plt.colorbar()
        plt.subplot(244)
        plt.title('i1')
        plt.imshow(i1_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.subplot(247)
        plt.title('Out-GT')
        plt.imshow(residue[var])
        plt.colorbar()

        plt.show()

    def sanity_check_regular_loss(self):
        '''
        train over ramdomly cropped patch from one triplets
        '''
        self.load_writer()
        self.load_model()

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)
        tran = transforms.Compose([
                                  Resize(self.input_size), 
                                  Normalize(mean=.5), 
                                  CustomPad((8, 0, 8, 0), 'circular'), 
                                  CustomPad((0, 8, 0, 8), 'zero', constant_values=0), 
                                  ])
        to_tensor = ToTensor()
        g_randcroup = GroupRandomCrop(self.crop_size, label_size=self.label_size)
        i0_in, i1_in, label_in = self.data_tr[20]

        i0_in = tran(i0_in)
        i1_in = tran(i1_in)
        label_in = tran(label_in)

        i0, i1, label = g_randcroup(i0_in, i1_in, label_in)
        i0 = to_tensor(i0)
        i1 = to_tensor(i1)
        i1_crop = i1[:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
        label = to_tensor(label)
        duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
        i1_crop = torch.unsqueeze(i1_crop, 0)
        label = torch.unsqueeze(label, 0)
        duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
        self.model.train()
        for iter_id in range(1):
            output = self.model(duo)
            loss = self.criterion(output + i1_crop, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))
            self.step += 1
        self.save()

        # visualize overfit result
        var=1
        self.model.eval()
        i0 = to_tensor(i0_in)
        i1 = to_tensor(i1_in)
        label = to_tensor(label_in)
        duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
        i0_crop = i0[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        i1_crop = i1[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        duo_crop = torch.unsqueeze(torch.cat([i0_crop, i1_crop], dim=0), 0)

        # duo, i0_crop, i1_crop = duo.to(self.device), i0_crop.to(self.device), i1_crop.to(self.device)
        duo, duo_crop, label, i0_crop, i1_crop = duo.to(self.device), duo_crop.to(self.device), label.to(self.device), i0_crop.to(self.device), i1_crop.to(self.device)
        with torch.no_grad():
            output = self.model(duo)
            out = output[0] + i1_crop
            # pdb.set_trace()
            residue = out - label[:,8:-8,8:-8]
        norm = Normalize()
        self.writer.add_image('i0', norm(i0[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('gt', norm(label[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('synthesized', norm(out)[var], dataformats='HW')
        self.writer.add_image('i1_cropped', norm(i1_crop)[var], dataformats='HW')
        plt.subplot(241)
        plt.imshow(i0_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.title('i0')
        plt.subplot(242)
        plt.title('GT')
        plt.imshow(label_in[8:-8,8:-8,var])   
        plt.colorbar() 
        plt.subplot(243)
        plt.title('Out')
        plt.imshow(out[var].to('cpu'))
        plt.colorbar()
        plt.subplot(244)
        plt.title('i1')
        plt.imshow(i1_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.subplot(247)
        plt.title('Out-GT')
        plt.imshow(residue[var].to('cpu'))
        plt.colorbar()

        plt.show()


    def sanity_check_randcrop_interpo_loss(self):
        '''
        train over ramdomly cropped patch from one triplets
        1. change input size to 32
        2. change output layer feature to 8
        '''
        self.load_writer()
        self.load_model()

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False)
        tran = transforms.Compose([
                                  Resize(self.input_size), 
                                  Normalize(mean=.5), 
                                  CustomPad((8, 0, 8, 0), 'circular'), 
                                  CustomPad((0, 8, 0, 8), 'zero', constant_values=0), 
                                  ])
        to_tensor = ToTensor()
        i0, i1, label = self.data_tr[20]

        i0_in = tran(i0)
        i1_in = tran(i1)
        label_in = tran(label)

        self.model.train()
        for iter_id in range(1000):
            i0 = to_tensor(i0_in)
            i1 = to_tensor(i1_in)
            label = to_tensor(label_in)
            duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
            i0_crop = i0[:,8:-8,8:-8]
            i1_crop = i1[:,8:-8,8:-8]
            label = label[:,8:-8,8:-8]

            duo_crop = torch.unsqueeze(torch.cat([i0_crop, i1_crop], dim=0), 0)
            i0_crop = torch.unsqueeze(i0_crop, 0)
            i1_crop = torch.unsqueeze(i1_crop, 0)
            label = torch.unsqueeze(label, 0)
            duo, duo_crop, label, i0_crop, i1_crop = duo.to(self.device), duo_crop.to(self.device), label.to(self.device), i0_crop.to(self.device), i1_crop.to(self.device)
            output = self.model(duo)

            two_img_to_concate = duo_crop * output
            _img1, _img2 = torch.split(two_img_to_concate, 4, dim=1)
            loss = self.criterion(_img1 + _img2, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))
            self.step += 1
        self.save()

        # visualize overfit result
        var=1
        self.model.eval()
        i0 = to_tensor(i0_in)
        i1 = to_tensor(i1_in)
        label = to_tensor(label_in)
        duo = torch.unsqueeze(torch.cat([i0, i1], dim=0), 0)
        i0_crop = i0[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        i1_crop = i1[:,8:self.input_size[0]+8,8:self.input_size[0]+8]
        duo_crop = torch.unsqueeze(torch.cat([i0_crop, i1_crop], dim=0), 0)
        duo, duo_crop, label, i0_crop, i1_crop = duo.to(self.device), duo_crop.to(self.device), label.to(self.device), i0_crop.to(self.device), i1_crop.to(self.device)
        with torch.no_grad():
            output = self.model(duo)
            two_img_to_concate = duo_crop * output
            _img1, _img2 = torch.split(two_img_to_concate, 4, dim=1)
            out = _img1 + _img2
            residue = out - label[:,8:-8,8:-8]
            out = out[0]
            residue = residue[0]
        norm = Normalize()
        self.writer.add_image('i0', norm(i0[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('gt', norm(label[:,8:-8,8:-8])[var], dataformats='HW')
        self.writer.add_image('synthesized', norm(out)[var], dataformats='HW')
        self.writer.add_image('i1_cropped', norm(i1_crop)[var], dataformats='HW')
        plt.subplot(241)
        plt.imshow(i0_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.title('i0')
        plt.subplot(242)
        plt.title('GT')
        plt.imshow(label_in[8:-8,8:-8,var])   
        plt.colorbar() 
        plt.subplot(243)
        plt.title('Out')
        plt.imshow(out[var].to('cpu'))
        plt.colorbar()
        plt.subplot(244)
        plt.title('i1')
        plt.imshow(i1_in[8:-8,8:-8,var])
        plt.colorbar()
        plt.subplot(247)
        plt.title('Out-GT')
        plt.imshow(residue[var].to('cpu'))
        plt.colorbar()

        plt.show()


    def sanity_check_no_randcrop(self):
        '''
        Sanity check without random crop
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


    def train(self, valid = True):
        '''
        Train the model
        epoch_loss: total train loss
        '''
        print("\ntrain:")
        self.model.train()
        # pdb.set_trace()

        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            # Only cut i1 for err calc
            i1_crop = i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            # Concatenate two input imgs in NCHW format
            duo = torch.cat([i0, i1], dim=1)
            duo, label, i1_crop = duo.to(self.device), label.to(self.device), i1_crop.to(self.device)
            output = self.model(duo)
            # print(self.model.state_dict().keys())
            # Avg per img loss: Err = f(I0,I1) + I1 - I0.5
            loss = self.criterion(output + i1_crop, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {:.4f}".format(self.step, loss.item()))

            if self.step % 10 == 0 and valid == True:
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

        # Construct network grgh
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
            train_result = self.train(valid = self.valid_required)
            # print("Epoch {} loss: {}".format(epoch, train_result))
            # accuracy = max(accuracy, valid_result[1])
            # test_result = self.test()
            # accuracy = max(accuracy, test_result[1])
            # Save checkpoint periodically
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(accuracy, epoch)

        print("Training finished")
        self.writer._get_file_writer().flush()
        self.save()
        # Remove the checkpoint when training finished and the model is saved
        print("Checkpoint removed upon training complete")
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)


    def test_single(self, triplet_id = None, step_diff = None, audience='astro'):
        '''
        Visualize using trained model
        triplet_id: triplet index to use, default will be random
        step_diff: a tuple of (min_step_diff, max_step_diff)
        audience: 'astro' or 'cs' for different visualization arrangement
        '''
        var = 1
        n_row = 5
        # Randomly choose triplet if not given
        if not triplet_id:
            triplet_id = np.random.randint(0, len(data_tr)-1)
        if step_diff:
            self.min_step_diff = step_diff[0]
            self.max_step_diff = step_diff[1]

        self.load_writer()
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
        # Fetch triplets and transform
        i0, i1, label, info_dict = data_va[triplet_id]

        # tran = transforms.Compose([
        #                           # Crop((0, 0), (440, 1024)), 
        #                           Resize(self.input_size), 
        #                           Normalize(mean=.5), 
        #                           # Resize((55, 128)), 
        #                           # Resize((32, 32)), 
        #                           ])
        # i0 = tran(i0)
        # i1 = tran(i1)
        # label = tran(label)
        resize = Resize(self.input_size)
        i0_sized = resize(i0)
        i1_sized = resize(i1)
        label_sized = resize(label)

        norm = Normalize(mean=.5)
        i0_normed = norm(i0_sized)
        i1_normed = norm(i1_sized)
        label_normed = norm(label_sized)

        # self.writer.add_images('triplet', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
        self.writer.add_images('i0', i0[:,:,var], dataformats='HW')
        self.writer.add_images('i1', i1[:,:,var], dataformats='HW')
        self.writer.add_images('label', label[:,:,var], dataformats='HW')
        pad = transforms.Compose([CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                                  CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0)])
        i0_padded = pad(i0_normed)
        i1_padded = pad(i1_normed)

        to_tensor = ToTensor()
        i0_normed = to_tensor(i0_normed)
        i0_padded = to_tensor(i0_padded)
        i1_normed = to_tensor(i1_normed)
        i1_padded = to_tensor(i1_padded)
        label_normed = to_tensor(label_normed)

        # Setup network
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

        # Run the network with input
        self.model.eval()
        with torch.no_grad():
            duo = torch.cat([i0_padded, i1_padded], dim=0)
            duo = torch.unsqueeze(duo, 0)
            output = self.model(duo)
            out = output[0] + i1_normed
            residue = out - label_normed
            original_diff = i1_normed - label_normed

        # Visualize and add to summary
        # Prepare data
        out = out[var]
        residue = residue[var]
        original_diff = original_diff[var]
        i0_sized = resize(i0)[:,:,var]
        i1_sized = resize(i1)[:,:,var]
        label_sized = resize(label)[:,:,var]
        # default dpi 6.4, 4.8
        # plt.figure(figsize=(20, 4), dpi=200).
        if audience == 'astro':
            plt.subplot(241)
            plt.title('i0')
            plt.imshow(i0_sized)
            plt.colorbar()
            plt.subplot(242)
            plt.title('i1')
            plt.imshow(i1_sized)
            plt.colorbar()
            plt.subplot(243)
            plt.title('GT')
            plt.imshow(label_sized) 
            plt.colorbar() 
            plt.subplot(244)
            plt.title('Out')
            plt.imshow(out)
            plt.colorbar()
            plt.subplot(248)
            plt.title('Residue(Out-Normalize(GT))')
            plt.imshow(residue)
            plt.colorbar()
        elif audience == 'cs':
            plt.subplot(241)
            plt.title('GT')
            plt.imshow(label_normed[var])   
            plt.colorbar() 
            plt.subplot(242)
            plt.title('Out')
            plt.imshow(out)
            plt.colorbar()
            plt.subplot(243)
            plt.title('i1_cropped')
            plt.imshow(i1_normed[var])
            plt.colorbar()
            # Calculate max/min of resudue & original difference together
            vmax = max(torch.max(residue), torch.max(original_diff))
            vmin = min(torch.min(residue), torch.min(original_diff))
            plt.subplot(244)
            plt.title('Out-GT(rescaled)')
            plt.imshow(residue, vmin = vmin, vmax = vmax)
            plt.colorbar()
            plt.subplot(247)
            plt.title('Out-GT')
            plt.imshow(residue)
            plt.colorbar()
            plt.subplot(248)
            plt.title('i1_cropped-GT(rescaled)')
            plt.imshow(original_diff, vmin = vmin, vmax = vmax)
            plt.colorbar()

        norm = Normalize()
        self.writer.add_image('residue', norm(residue), dataformats='HW')
        self.writer.add_image('synthetic', norm(out), dataformats='HW')
        self.writer.add_image('resized_i0', i0_normed[var], dataformats='HW')
        self.writer.add_image('resized_i1', i1_normed[var], dataformats='HW')
        self.writer.add_image('resized_label', label_normed[var], dataformats='HW')


        # plt.savefig("a.png")
        print("triplet id: ", triplet_id)
        print("disk name: ", info_dict["disk_name"])
        print("Image_t0_idx: {}, Image_t1_idx: {}, label_idx: {}".format(info_dict["img1_idx"], info_dict["img2_idx"], info_dict["label_idx"]))
        print("Residue sum: ", torch.sum(torch.abs(residue)))
        print("Original diff: ", torch.sum(torch.abs(original_diff)))
        plt.show()
