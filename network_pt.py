# Neural Network Class

import pdb
import os
import shutil
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from model import ResNet, UNet
from pathlib import Path
from util.data_util import get_stats
from util.transform import *
from dataset.Astrodata import Astrodata

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
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
        self.report_freq = config.report_freq
        self.input_size = config.input_size
        self.crop_size = config.crop_size
        self.label_size = config.label_size
        self.nvar = config.nvar
        self.num_workers = config.num_workers
        self.valid_required = True
        self.pin_memory = False
        self.non_blocking = False 
        
        # For debug on Windows
        if os.name == 'nt':
            self.non_blocking = False
            self.pin_memory = False
            self.data_dir = str(config.h5_dir_win)
            self.batch_size = 5
            self.valid_required = True
            self.epochs = 50
            self.min_step_diff = None
            self.num_workers = 0
            self.report_freq = 1
            self.checkpoint_freq = 1
            self.lr = 0.0001

        # Inferenced parameter
        self.tr_data_dir = Path(self.data_dir + "_tr.h5")
        self.va_data_dir = Path(self.data_dir + "_va.h5")
        # Sizes to crop the input img
        self.ltl = (int(0.5 * (self.crop_size[0] - self.label_size[0])), int(0.5 * (self.crop_size[1] - self.label_size[1])))
        self.lbr = (self.ltl[0] + self.label_size[0], self.ltl[1] + self.label_size[1])
        # Global step
        self.step = 0


    def load_writer(self):
        '''
        Load the tensorboard writter
        '''
        self.writer = SummaryWriter(logdir=str(self.log_dir), flush_secs=120)


    def load_data(self):
        '''
        Prepare train/test data
        '''
        mean, std = get_stats(self.tr_data_dir, self.va_data_dir, self.nvar)
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
                 # Resize((self.input_size)), # Done in .5py
                 # LogPolartoPolar(), # Use polar data instead, too expensive
                 CustomPad((math.ceil((self.crop_size[1] - self.label_size[1])/2), 0, math.ceil((self.crop_size[1] - self.label_size[1])/2), 0), 'circular'), 
                 CustomPad((0, math.ceil((self.crop_size[0] - self.label_size[0])/2), 0, math.ceil((self.crop_size[0] - self.label_size[0])/2)), 'zero', constant_values=0), 
                 GroupRandomCrop(self.crop_size, label_size=self.label_size), 
                 Normalize(mean, std),
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
                            transforms = trans) # RandomCrop is group op

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
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
                                       num_workers= self.num_workers, 
                                       # sampler = train_sampler, 
                                       shuffle = True, 
                                       pin_memory = self.pin_memory)
        self.valid_loader = DataLoader(self.data_va, 
                                       batch_size = self.batch_size, 
                                       num_workers = self.num_workers, 
                                       # sampler = valid_sampler, 
                                       shuffle = True, 
                                       pin_memory = self.pin_memory)


    def load_model(self):
        '''
        Load the torch model
        '''
        if self.cuda:
            print("Using GPU")
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            print("Using CPU")
            self.device = torch.device('cpu')

        # import IPython
        # IPython.embed()
        # self.model = ResNet().to(self.device)
        self.model = UNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 10], gamma=0.5)
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
                            max_step_diff = self.max_step_diff
                            )

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff,
                            )
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
                            max_step_diff = self.max_step_diff
                            )

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = 40, 
                            max_step_diff = self.max_step_diff
                            )
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
                            max_step_diff = self.max_step_diff
                            )

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff
                            )
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


    def train(self):
        '''
        Train the model
        epoch_loss: total train loss
        '''
        print("\ntrain:")
        self.model.train()
        train_loss = 0
        n_batch = 0
        # torch.cuda.synchronize()
        start = time.time()
        # start0 = time.time()
        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            # Concatenate two imgs
            # torch.cuda.synchronize()
            # time0 = time.time()
            # print("batch load: ", time0 - start0)
            label, _i0, _i1 = label.to(self.device, non_blocking = self.non_blocking), i0.to(self.device, non_blocking = self.non_blocking), i1.to(self.device, non_blocking = self.non_blocking)
            # torch.cuda.synchronize()
            # time1 = time.time()
            # print("transfer: ",time1 - time0)
            # Only cut i1 for err calc
            i1_crop = _i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            # torch.cuda.synchronize()
            # time2 = time.time()
            # print("crop: ", time2 - time1)
            duo = torch.cat([_i0, _i1], dim=1)
            # torch.cuda.synchronize()
            # time3 = time.time()
            # print("cat: ", time3 - time2)
            self.optimizer.zero_grad()
            # torch.cuda.synchronize()
            # time4=time.time()
            # print("zero_grad: ", time4 - time3)
            output = self.model(duo)
            # torch.cuda.synchronize()
            # time5=time.time()
            # print("fwd pass: ", time5 - time4)
            # print(self.model.state_dict().keys())
            loss = self.criterion(output + i1_crop, label)
            # torch.cuda.synchronize()
            # time6 = time.time()
            # print("loss calc: ", time6 - time5)
            loss.backward()
            # torch.cuda.synchronize()
            # time7 = time.time()
            # print("back prop: ", time7 - time6)
            self.optimizer.step()
            # torch.cuda.synchronize()
            # time8 = time.time()
            # print("optimizer step: ", time8 - time7)
            train_loss += loss.item()
            self.step += 1
            n_batch += 1 
            print("step{}, loss: {:.4f}".format(self.step, loss.item()))
            # torch.cuda.synchronize()
            # time9 = time.time()
            # print("last step: {}\n".format(time9 - time8))
            # start0 = time.time()

        print("Time {} sec".format(time.time() - start))

        return train_loss / n_batch


    def valid(self):
        '''
        Test the accuracy of the current model parameters
        '''
        print("\nvalid:")
        valid_loss = 0
        n_batch = 0

        with torch.no_grad():
            for b_id, (i0, i1, label) in enumerate(self.valid_loader):
                label, _i0, _i1 = label.to(self.device, non_blocking = self.non_blocking), i0.to(self.device, non_blocking = self.non_blocking), i1.to(self.device, non_blocking = self.non_blocking)
                # Only cut i1 for err calc
                i1_crop = _i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
                # Concatenate two input imgs in NCHW format
                duo = torch.cat([_i0, _i1], dim=1)
                output = self.model(duo)
                # L1 Loss
                loss = self.criterion(output + i1_crop, label)
                valid_loss += loss.item()
                n_batch += 1
                print("batch{}, loss: {}".format(b_id, loss.item()))

        return valid_loss / n_batch


    def load_checkpoint(self):
        '''
        Load the model from checkpoing
        '''
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy = checkpoint['accuracy']
        self.step = checkpoint['step'] + 1 # +1 for the next step

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
        self.load_model()
        self.load_data()
        self.load_writer()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of network parameters: ", params)
        tensor_list = list(self.model.state_dict().items())
        for layer_tensor_name, tensor in tensor_list:
            print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))

        # Construct network grgh
        # sample_input=(torch.rand(1, 8, self.crop_size[0], self.crop_size[1]))
        # self.writer.add_graph(model = ResNet(), input_to_model=sample_input)

        accuracy = 0
        start_epoch = 1

        # Check if previous checkpoint/model file exists
        if Path(self.checkpoint).exists():
            accuracy, epoch = self.load_checkpoint()
            start_epoch = epoch + 1
            print("Checkpoint loaded starting from epoch {} step {}".format(start_epoch, self.step))

        # Iterate through epochs
        for epoch in range(start_epoch, self.epochs + 1):
            # self.scheduler.step(epoch)
            print("\n===> epoch: {}/{}".format(epoch, self.epochs))
            train_result = self.train()
            print("Epoch {} loss: {}".format(epoch, train_result))
            # test_result = self.test()
            if epoch % self.report_freq == 0:
                if self.valid_required:
                    self.model.eval()
                    valid_result = self.valid()
                    self.writer.add_scalar('Valid/Loss', valid_result, self.step)
                    print("Validation loss: {}".format(valid_result))
                self.model.train()
                self.writer.add_scalar('Train/Loss', train_result, self.step)
            self.writer._get_file_writer().flush()
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(accuracy, epoch)

        print("Training finished")
        self.writer._get_file_writer().flush()
        self.save()

        # # Remove the checkpoint when complete and the model is saved
        # if os.path.exists(self.checkpoint):
        #     os.remove(self.checkpoint)
        # print("Checkpoint removed upon training complete")


    def test_single(self, triplet_id = None, step_diff = None, audience = 'astro', dataset = 'va'):
        '''
        Visualize using trained model
        triplet_id: triplet index to use, default will be random
        step_diff: a tuple of (min_step_diff, max_step_diff)
        audience: 'astro' or 'cs' for different visualization arrangement
        '''
        var = 1
        n_row = 5
        if step_diff:
            self.min_step_diff = step_diff[0]
            self.max_step_diff = step_diff[1]

        self.load_writer()
        data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            verbose = True) # RandomCrop is group op

        data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            verbose = True) # RandomCrop is group op
        # Randomly choose triplet if not given
        if not triplet_id and triplet_id != 0:
            triplet_id = np.random.randint(0, len(data_tr)-1)
        # Fetch triplets and transform
        if dataset == 'tr':
            i0, i1, label, info_dict = data_tr[triplet_id]
        else:
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

        # resize = Resize(self.input_size) # Done in .5py
        # i0_sized = resize(i0)
        # i1_sized = resize(i1)
        # label_sized = resize(label)
        mean, std = get_stats(self.tr_data_dir, self.va_data_dir, self.nvar)
        norm = Normalize(mean, std)
        i0_normed = norm(np.array(i0))
        i1_normed = norm(np.array(i1))
        label_normed = norm(np.array(label))
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
        elif Path(self.checkpoint).exists():
            print("Model file does not exists, trying checkpoint")
            self.model = ResNet().to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.load_checkpoint()
        else:
            print("No model.pth or checkpoint.tar found, exiting.")
            exit(1)

        # Run the network with input
        self.model.eval()
        with torch.no_grad():
            duo = torch.cat([i0_padded, i1_padded], dim=0)
            duo = torch.unsqueeze(duo, 0)
            output = self.model(duo)
            out = output[0] + i1_normed
            residue = out - label_normed

        # Visualize and add to summary
        pick = chan(var)

        out_unormed = pick(out) * std[var] + mean[var]
        residue_unormed = np.array(out_unormed) - pick(label)
        original_diff = i0 - label
        original_diff = pick(original_diff)
        i0 = pick(i0)
        i1 = pick(i1)
        label = pick(label)
        # plt.figure(figsize=(20, 4), dpi=200) # default dpi 6.4, 4.8
        if audience == 'astro':
            plt.subplot(251)
            plt.title('i0')
            plt.imshow(i0)
            plt.colorbar()
            plt.subplot(252)
            plt.title('i1')
            plt.imshow(i1)
            plt.colorbar()
            plt.subplot(253)
            plt.title('GT')
            plt.imshow(label) 
            plt.colorbar() 
            plt.subplot(254)
            plt.title('Out')
            plt.imshow(out_unormed)
            plt.colorbar()
            plt.subplot(255)
            plt.title('Residue')
            plt.imshow(residue_unormed)
            plt.colorbar()

            plt.subplot(256)
            plt.title('i0_normed')
            plt.imshow(pick(i0_normed))
            plt.colorbar()
            plt.subplot(257)
            plt.title('i1_normed')
            plt.imshow(pick(i1_normed))
            plt.colorbar()
            plt.subplot(258)
            plt.title('GT_normed')
            plt.imshow(pick(label_normed))
            plt.colorbar()
            plt.subplot(259)
            plt.title('Out_normed')
            plt.imshow(pick(out))
            plt.colorbar()
            plt.subplot(2, 5, 10)
            plt.title('Residue_normed')
            plt.imshow(pick(residue))
            plt.colorbar()
        elif audience == 'cs':
            plt.subplot(241)
            plt.title('GT')
            plt.imshow(label_normed[var])   
            plt.colorbar() 
            plt.subplot(242)
            plt.title('Out')
            plt.imshow(out_unormed)
            plt.colorbar()
            plt.subplot(243)
            plt.title('i1_cropped')
            plt.imshow(pick(i1_normed))
            plt.colorbar()
            # Calculate max/min of resudue & original difference together
            vmax = max(np.amax(residue_unormed), np.amax(original_diff))
            vmin = min(np.amin(residue_unormed), np.amin(original_diff))
            plt.subplot(244)
            plt.title('Out-GT(rescaled)')
            plt.imshow(residue_unormed, vmin = vmin, vmax = vmax)
            plt.colorbar()
            plt.subplot(247)
            plt.title('Out-GT')
            plt.imshow(residue_unormed)
            plt.colorbar()
            plt.subplot(248)
            plt.title('i1_cropped-GT(rescaled)')
            plt.imshow(original_diff, vmin = vmin, vmax = vmax)
            plt.colorbar()

        self.writer.add_image('residue', residue_unormed + 0.5, dataformats='HW')
        self.writer.add_image('synthetic', out_unormed + 0.5, dataformats='HW')
        self.writer.add_image('resized_i0', i0, dataformats='HW')
        self.writer.add_image('resized_i1', i1, dataformats='HW')
        self.writer.add_image('resized_label', label, dataformats='HW')
        self.writer._get_file_writer().flush()
        
        # plt.savefig("a.png")
        print("triplet id: ", triplet_id)
        print("disk name: ", info_dict["disk_name"])
        print("Image_t0_idx: {}, Image_t1_idx: {}, label_idx: {}".format(info_dict["img1_idx"], info_dict["img2_idx"], info_dict["label_idx"]))
        print("Residue sum: ", np.sum(np.abs(residue_unormed)))
        print("Original diff: ", np.sum(np.abs(original_diff)))
        print("PSNR: ", self.psnr(label, (np.asarray(out_unormed) + 0.5), 1))
        plt.show()



    def psnr(self, orig, noisy, max_possible=1):
        '''
        orig: original img
        noisy: synthesized img
        max_possible: max 
        Calculate PSNR
        '''
        mse = np.mean((orig - noisy) ** 2)
        if mse == 0:
            return "Inf"
        PIXEL_MAX = max_possible

        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


    def clean_up(self):
        '''
        Clean up the network outputs and summary
        '''
        if Path(self.model_dir).exists():
            os.remove(self.model_dir)
        if Path(self.checkpoint).exists():
            os.remove(self.checkpoint)
        if Path('output').exists():
            shutil.rmtree("output")
        if Path('tmp').exists():
            shutil.rmtree("tmp") 
        os.mkdir(str(self.log_dir))
        print("Clean up finished")
