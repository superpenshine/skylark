# Neural Network Class

import pdb
import os
import shutil
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from model import ResNet, UNet, UNet2
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


def psnr(orig, noisy, max_possible=1):
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
            self.valid_required = True
            self.data_dir = str(config.h5_dir_win)
            self.batch_size = 1
            self.epochs = 3000
            self.min_step_diff = 74
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
                 # For Cropped input
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
        self.model = ResNet().to(self.device)
        # self.model = UNet().to(self.device)
        # self.model = UNet2().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 10], gamma=0.5)
        self.criterion = MSELoss().to(self.device) # set reduction=sum, or too smal to see
        # self.criterion = L1Loss().to(self.device)


    def train(self):
        '''
        Train the model
        epoch_loss: total train loss
        '''
        print("\ntrain:")
        self.model.train()
        train_loss = 0
        n_batch = 0
        start = time.time()
        for b_id, (i0, i1, label) in enumerate(self.train_loader):
            i0 = torch.unsqueeze(i0[:,1], 1)
            i1 = torch.unsqueeze(i1[:,1], 1)
            label = torch.unsqueeze(label[:,1], 1)
            label, _i0, _i1 = label.to(self.device, non_blocking = self.non_blocking), i0.to(self.device, non_blocking = self.non_blocking), i1.to(self.device, non_blocking = self.non_blocking)
            i1_crop = _i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]] 
            duo = torch.cat([_i0, _i1], dim=1)
            duo = torch.reshape(duo, (self.batch_size, 2*self.crop_size[0], 1, -1))
            self.optimizer.zero_grad()
            output = self.model(duo)
            output = torch.reshape(output, (self.batch_size, 1, self.label_size[0], -1))
            loss = self.criterion(output + i1_crop, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.step += 1
            n_batch += 1 
            print("step{}, loss: {:.4f}".format(self.step, loss.item()))

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
                i0 = torch.unsqueeze(i0[:,1], 1)
                i1 = torch.unsqueeze(i1[:,1], 1)
                label = torch.unsqueeze(label[:,1], 1)
                label, _i0, _i1 = label.to(self.device, non_blocking = self.non_blocking), i0.to(self.device, non_blocking = self.non_blocking), i1.to(self.device, non_blocking = self.non_blocking)
                # Only cut i1 for err calc
                i1_crop = _i1[:,:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
                # Concatenate two input imgs in NCHW format
                duo = torch.cat([_i0, _i1], dim=1)
                duo = torch.reshape(duo, (self.batch_size, 2*self.crop_size[0], 1, -1))
                output = self.model(duo)
                output = torch.reshape(output, (self.batch_size, 1, self.label_size[0], -1))
                # MSE Loss
                loss = self.criterion(output + i1_crop, label)
                valid_loss += loss.item()
                n_batch += 1
                print("batch{}, loss: {:.4f}".format(n_batch, loss.item()))

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
            print("Epoch {} loss: {:.4f}".format(epoch, train_result))
            # test_result = self.test()
            if epoch % self.report_freq == 0:
                if self.valid_required:
                    self.model.eval()
                    valid_result = self.valid()
                    self.writer.add_scalar('Valid/Loss', valid_result, self.step)
                    print("Validation loss: {:.4f}".format(valid_result))
                self.model.train()
                self.writer.add_scalar('Train/Loss', train_result, self.step)
            self.writer._get_file_writer().flush()
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(accuracy, epoch)

        print("Training finished")
        self.writer.close()
        self.save()

        # # Remove the checkpoint when complete and the model is saved
        # if os.path.exists(self.checkpoint):
        #     os.remove(self.checkpoint)
        # print("Checkpoint removed upon training complete")


    def test_single(self, triplet_id = None, step_diff = None, audience = 'normal', dataset = 'va'):
        '''
        Visualize using trained model
        triplet_id: triplet index to use, default will be random
        step_diff: a tuple of (min_step_diff, max_step_diff)
        audience: 'normal' or 'pipeline' for different visualization arrangement
        '''
        var = 1
        n_row = 5
        pick = chan(var)
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

        # Load network to cpu
        device = torch.device('cpu')
        if Path(self.model_dir).exists():
            self.load(map_location=device)
        elif Path(self.checkpoint).exists():
            print("Model file does not exists, trying checkpoint")
            self.model = ResNet().to(device)
            # self.model = UNet().to(device)
            # self.model = UNet2().to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.load_checkpoint()
        else:
            print("No model.pth or checkpoint.tar found, exiting.")
            exit(1)

        # Run the network with input
        self.model.eval()
        with torch.no_grad():
            duo = torch.cat([i0_padded, i1_padded], dim=0)
            duo = torch.reshape(duo, (self.batch_size, 8*self.crop_size[0], 1, -1))
            # duo = torch.unsqueeze(duo, 0)
            output = self.model(duo)
            output = torch.reshape(output, (self.batch_size, 4, self.label_size[0], -1))
            out = output[0] + i1_normed
            residue = out - label_normed

        # Visualize and add to summary
        output = output[0]
        out_unormed = pick(out) * pick(std) + pick(mean)
        residue_unormed = np.array(out_unormed) - pick(label)
        original_diff = i0 - label
        original_diff = pick(original_diff)
        i0 = pick(i0)
        i1 = pick(i1)
        label = pick(label)
        output = pick(output)
        i0_padded = pick(i0_padded)
        i1_padded = pick(i1_padded)
        # plt.figure(figsize=(20, 4), dpi=200) # default dpi 6.4, 4.8
        if audience == 'normal':
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
            # Second row
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
        elif audience == 'pipeline':
            plt.subplot(131)
            plt.title('i0_padded')
            plt.imshow(i0_padded)
            plt.colorbar()
            plt.subplot(132)
            plt.title('i1_padded')
            plt.imshow(i1_padded)
            plt.colorbar()
            plt.subplot(133)
            plt.title('Network_immediate_output')
            plt.imshow(output)
            plt.colorbar()
        else:
            raise ValueError("Unknown audience")

        # self.writer.add_images('triplet', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
        self.writer.add_image('img4_i0padded', pad(i0), dataformats='HW')
        self.writer.add_image('img5_i1padded', pad(i1), dataformats='HW')
        self.writer.add_image('img2_label', label, dataformats='HW')
        self.writer.add_image('img7_network_output', output + torch.min(output), dataformats='HW')
        self.writer.add_image('img6_residue', residue_unormed + np.amin(residue_unormed), dataformats='HW')
        self.writer.add_image('img3_synthetic', out_unormed + torch.min(out_unormed), dataformats='HW')
        self.writer.add_image('img0_resized_i0', i0, dataformats='HW')
        self.writer.add_image('img1_resized_i1', i1, dataformats='HW')
        self.writer.close()
        
        # plt.savefig("a.png")
        print("triplet id: ", triplet_id)
        print("disk name: ", info_dict["disk_name"])
        print("Image_t0_idx: {}, Image_t1_idx: {}, label_idx: {}".format(info_dict["img1_idx"], info_dict["img2_idx"], info_dict["label_idx"]))
        print("Residue sum: ", np.sum(np.abs(residue_unormed)))
        print("Original diff: ", np.sum(np.abs(original_diff)))
        print("PSNR: ", psnr(label, (np.asarray(out_unormed) + 0.5), 1))
        plt.show()


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
