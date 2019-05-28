# Neural Network Class

import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18, ResNet
from pathlib import Path
from util.transform import CustomPad, GroupRandomCrop, ToTensor
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
        self.label_size = config.label_size

        # Sizes to crop the input img
        self.ltl = (int(0.5 * (self.input_size[0] - self.label_size[0])), int(0.5 * (self.input_size[1] - self.label_size[1])))
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
        trans = [CustomPad((int((self.input_size[1]-1)*0.5), 0, int((self.input_size[1]-1)*0.5), 0), 'circular'), 
                 CustomPad((0, int((self.input_size[0]-1)*0.5), 0, int((self.input_size[0]-1)*0.5)), 'zero', constant_values=0), 
                 # CustomPad((0, int((self.input_size[1]-1)*0.5), 0, int((self.input_size[1]-1)*0.5)), 'circular'), 
                 # CustomPad((int((self.input_size[0]-1)*0.5), 0, int((self.input_size[0]-1)*0.5), 0), 'zero', constant_values=0), 
                 GroupRandomCrop(self.input_size, label_size=self.label_size), 
                 ToTensor()
                 # transforms.ToPILImage(), 
                 # transforms.Resize((128, 128)), # Using RandomCrop now
                 # transforms.Pad((64, 0, 64, 0), 'constant'), # only supports RGB
                 # transforms.RandomCrop(128, 128), # won't work since we want random crop at the same posion of the three images
                 # transforms.ToTensor()
                 ]

        self.data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            transforms = trans, 
                            group_trans_id = [2]) # RandomCrop is group op

        self.data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False, 
                            transforms = trans, 
                            group_trans_id = [2]) # RandomCrop is group op
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
        self.criterion = L1Loss(reduction='sum').to(self.device)


    def single_batch_train(self):
        '''
        Overfit over one batch
        '''
        output2 = None

        print("\ntrain:")
        self.model.train()
        for b_id, (i0, i1, label) in enumerate(self.train_loader):
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
            self.optimizer.zero_grad()
            output = self.model(duo)

            # Avg per img loss: Err = f(I0,I1) + I1 - I0.5
            loss = self.criterion(output + i1_crop, label)
            loss.backward()
            self.optimizer.step()
            print("step{}, loss: {}".format(self.step, loss.item()))

            if self.step % 5 == 0:
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
            for b_id, (i0, i1, label) in enumerate(self.train_loader):
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
                if b_id > 20:
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

        sample_input=(torch.rand(1, 8, self.input_size[0], self.input_size[1]))
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


    # def test_full(self):
    #     self.load_writer()
    #     var = 1
    #     tran = transforms.Resize((32, 32))
    #     data_tr = Astrodata(self.tr_data_dir, 
    #                         min_step_diff = self.min_step_diff, 
    #                         max_step_diff = self.max_step_diff, 
    #                         rtn_log_grid = False) # RandomCrop is group op

    #     data_va = Astrodata(self.va_data_dir, 
    #                         min_step_diff = self.min_step_diff, 
    #                         max_step_diff = self.max_step_diff, 
    #                         rtn_log_grid = False) # RandomCrop is group op
    #     i0, i1, label = data_tr[np.random.randint(0, len(data_tr)-1)]
    #     self.writer.add_images('step0/raw/i0_label_i1', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
    #     tran3 = ToTensor()
    #     i0 = tran3(i0)
    #     i1 = tran3(i1)
    #     label = tran3(label)
    #     device = torch.device('cpu')
    #     self.load(map_location=device)
    #     self.model.eval()
    #     with torch.no_grad():
    #         i1_crop = i1[:,8:1016,8:1016]
    #         duo = torch.cat([i0, i1], dim=0)
    #         duo = torch.unsqueeze(duo, 0)
    #         output = self.model(duo)
    #         out = output[0] + i1_crop

    #     out = out[var]
    #     self.writer.add_image('step4/synthetic', out, dataformats='HW')

    #     i1_crop = i1_crop[var]
    #     self.writer.add_image('step5/i1_centercrop', i1_crop, dataformats='HW')


    def test_single(self):
        '''
        Test the model using single inputj, for illustration
        '''
        self.load_writer()
        group_trans_id = [2]
        tran0 = CustomPad((int((self.input_size[1]-1)*0.5), 0, int((self.input_size[1]-1)*0.5), 0), 'circular')
        tran1 = CustomPad((0, int((self.input_size[0]-1)*0.5), 0, int((self.input_size[0]-1)*0.5)), 'zero', constant_values=0)
        tran2 = GroupRandomCrop(self.input_size, label_size=self.label_size)
        tran3 = ToTensor()
        var = 1
        n_row = 5

        data_tr = Astrodata(self.tr_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op

        data_va = Astrodata(self.va_data_dir, 
                            min_step_diff = self.min_step_diff, 
                            max_step_diff = self.max_step_diff, 
                            rtn_log_grid = False) # RandomCrop is group op

        # Normalized triplet without transform
        i0, i1, label = data_tr[np.random.randint(0, len(data_tr)-1)]
        # plt.subplot(n_row, 3, 1)
        # plt.imshow(i0[:,:,var])
        # plt.subplot(n_row, 3, 2)
        # plt.imshow(label[:,:,var])
        # plt.subplot(n_row, 3, 3)
        # plt.imshow(i1[:,:,var])
        self.writer.add_images('step0/raw/i0_label_i1', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
        # self.writer.add_image('i0_raw', i0[:,:,var], dataformats='HW')
        # self.writer.add_image('gt_raw', label[:,:,var], dataformats='HW')
        # self.writer.add_image('i1_raw', i1[:,:,var], dataformats='HW')

        i0 = tran0(i0)
        i1 = tran0(i1)
        label = tran0(label)
        i0 = tran1(i0)
        i1 = tran1(i1)
        label = tran1(label)
        # plt.subplot(n_row, 3, 4)
        # plt.imshow(i0[:,:,var])
        # plt.subplot(n_row, 3, 5)
        # plt.imshow(label[:,:,var])
        # plt.subplot(n_row, 3, 6)
        # plt.imshow(i1[:,:,var])
        self.writer.add_images('step1/pad/i0_label_i1', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
        # self.writer.add_image('i0_pad', i0[:,:,var], dataformats='HW')
        # self.writer.add_image('gt_pad', label[:,:,var], dataformats='HW')
        # self.writer.add_image('i1_pad', i1[:,:,var], dataformats='HW')

        i0, i1, label = tran2(i0, i1, label)
        # plt.subplot(n_row, 3, 7)
        # plt.imshow(i0[:,:,var])
        # plt.subplot(n_row, 3, 8)
        # plt.imshow(label[:,:,var])
        # plt.subplot(n_row, 3, 9)
        # plt.imshow(i1[:,:,var])
        # self.writer.add_images('step2/randcrop/i0_label_i1', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')
        self.writer.add_image('step2.1/i0_randcrop', i0[:,:,var], dataformats='HW')
        self.writer.add_image('step2.2/gt_randcrop', label[:,:,var], dataformats='HW')
        self.writer.add_image('step2.3/i1_randcrop', i1[:,:,var], dataformats='HW')

        i0 = tran3(i0)
        i1 = tran3(i1)
        label = tran3(label)

        device = torch.device('cpu')
        self.load(map_location=device)
        self.model.eval()
        with torch.no_grad():
            i1_crop = i1[:,self.ltl[0]:self.lbr[0],self.ltl[1]:self.lbr[1]]
            duo = torch.cat([i0, i1], dim=0)
            duo = torch.unsqueeze(duo, 0)
            output = self.model(duo)
            out = output[0] + i1_crop
            residue = torch.abs(out - label)
            original_diff = torch.abs(i1_crop - label)

        original_diff = original_diff[var]
        # plt.subplot(n_row, 3, 10)
        # plt.imshow(original_diff)
        # self.writer.add_image('i1-gt', original_diff, dataformats='HW')
        residue = residue[var]
        # plt.subplot(n_row, 3, 11)
        # plt.imshow(residue)
        # self.writer.add_image('synthetic-gt', residue, dataformats='HW')
        # self.writer.add_images('step3/i1-gt_synthetic-gt', np.expand_dims(np.stack([i0[:,:,var], label[:,:,var], i1[:,:,var]]), 3), dataformats='NHWC')

        # test = np.array([[1.0, 1.0], [0.4, 0.7]])
        # plt.subplot(n_row, 3, 13)
        # plt.imshow(test, vmin=0, vmax=1)

        out = out[var]
        # plt.subplot(n_row, 3, 14)
        # plt.imshow(out)
        self.writer.add_image('step4/synthetic', out, dataformats='HW')

        i1_crop = i1_crop[var]
        # plt.subplot(n_row, 3, 15)
        # plt.imshow(i1_crop)
        self.writer.add_image('step5/i1_centercrop', i1_crop, dataformats='HW')

        # plt.show()
