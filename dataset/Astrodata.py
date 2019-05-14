# Custom dataset class


import h5py
import numpy as np
from torch.utils.data import Dataset

class Astrodata(Dataset):
    '''
    Astro dataset class
    ''' 

    def __init__(self, data_dir, min_step_diff = None, max_step_diff = None, rtn_log_grid = False, transform = None):
        '''
        transform: transformations to apply on imgs
        '''
        self.data = h5py.File(data_dir)
        self.d_names = list(self.data.keys())
        # Minus 1 to get rid of "log_grid" in each disk
        self.d_steps = [len(self.data[d_name].keys()) - 1 for d_name in self.d_names]
        self.min_step_diff = min_step_diff
        self.max_step_diff = max_step_diff
        self.rtn_log_grid = rtn_log_grid
        self.transform = transform

        self.calc_valid_step_diffs()
        self.step_diff_validity_check()

        if not self.min_step_diff:
            self.min_step_diff = 2

        self.calc_bounds()


    def __getitem__(self, idx):
        '''
        d: disk name
        l, h: Two input imgs for training
        m: label img
        '''
        d, l, h, m = self.mapping(idx)
        print(d, l, h, m)
        l = self.data[d][str(l)]
        h = self.data[d][str(h)]
        m = self.data[d][str(m)]

        # Transform to uint8 0-255, required by PIL module
        l = self.normalize(np.array(l))
        h = self.normalize(np.array(h))
        m = self.normalize(np.array(m))

        # Apply transforms
        if self.transform:
            l = self.transform(l)
            h = self.transform(h)
            m = self.transform(m)

        if self.rtn_log_grid:
            log_grid = np.asarray(self.data[d]["log_grid"])
            return log_grid, l, h, m
        return l, h, m


    def __len__(self):
        return sum(self.d_bounds)


    def normalize(self, ndarray):
        '''
        Normalize the ndarray's all channels to 0-255
        ndarray: numpy array in HWC
        '''
        n_chan = ndarray.shape[-1]
        for c_i in range(n_chan):
            c_max = np.amax(ndarray[:,:,c_i])
            ndarray[:,:,c_i] = ndarray[:,:,c_i] / c_max

        return (ndarray * 255).astype(np.uint8)



    def mapping(self, idx):
        '''
        Map the index to disk name, img pair's idx, and the label img
        idx: index with respect to the pair list

            ex: data{
                    disk0 = [step0, 1, 2, 3]
                    disk1 = [step0, 1, 2, 3, 4, 5]
                }
                pair_list0 = [(0, 2), (1, 3)]
                pair_list1 = [(0, 2), (0, 4), (2, 4), (3, 5), (0, 4), (1, 5)]
                label_img0 = [1, 2]
                label_img1 = [1, 2, 3, 4]

                input: idx = 4

                return: "disk1", 0, 4, 2
        '''
        for i, d_bound in enumerate(self.d_bounds):
            if idx >= d_bound:
                idx -= d_bound
                continue

            l, h, m = self.find_triplets(i, idx)
            return (self.d_names[i], l, h, m)


    def find_triplets(self, disk_idx, idx):
        '''
        Find 2 paired img idxes and the label img idx
        disk_idx: index of the disk
        idx: index with respect to the given disk

        return: triplets of (I_0_idx, I_1_idx, I_0.5_idx)

            ex: data{
                    disk0 = [step0, 1, 2, 3]
                    disk1 = [step0, 1, 2, 3, 4, 5]
                }
                disk0_step_2_pairs = [(0, 2), (1, 3)]
                disk1_step_2_pairs = [(0, 2), (1, 3), (2, 4), (3, 5)]
                disk1_step_4_pairs = [(0, 4), (1, 4)]
                label_img0 = [1, 2]
                label_img1 = [1, 2, 3, 4]

                input: disk_idx = 1, idx = 5
                
                Algorithm: 
                1. Skip disk0_step_2_pairs, by disk_idx = 1 
                2. Skip disk1_step_2_pairs since idx >=4
                3. idx -= 4, idx = 0
                4. idx < len(disk1_step_4_pairs), found I_0_idx
                5. Return 
                    (I_0_idx,
                     I_1_idx = I_0_idx + step_diff, 
                     I_0.5_idx =  I_0_idx + 0.5 * step_diff
                    )


        '''
        max_valid = self.max_valid_step_diffs[disk_idx]
        d_step = self.d_steps[disk_idx]

        if self.max_step_diff:
            max_valid = min(max_valid, self.max_step_diff)
        for step_diff in np.arange(self.min_step_diff, max_valid + 1, 2):
            pair_of_diff = d_step - step_diff
            if idx >= pair_of_diff:
                idx -= pair_of_diff
                continue
            l = idx
            h = l + step_diff
            m = int(l + 0.5 * step_diff)

            return l, h, m


    def calc_valid_step_diffs(self):
        '''
        For each disk, calculate the maximum number of valid
        step.
        '''
        self.max_valid_step_diffs = []

        for d_step in self.d_steps:
            # Set maximum possible step difference
            if d_step % 2 == 0:
                self.max_valid_step_diffs.append(d_step - 2)
            else:
                self.max_valid_step_diffs.append(d_step - 1)


    def step_diff_validity_check(self):
        '''
        Check the validity of min/max step difference
        '''
        if self.min_step_diff:
            if self.min_step_diff % 2 != 0:
                raise ValueError("Min_step_diff must be an even number.")

            if self.min_step_diff > max(self.max_valid_step_diffs):
                raise ValieError("Min_step_diff should be less or equal to {}".format(max(elf.max_valid_step_diffs)))

        if self.max_step_diff:
            if self.max_step_diff % 2 != 0:
                raise ValueError("Max_step_diff must be an even number.")

        if self.min_step_diff and self.max_step_diff:
            if self.min_step_diff > self.max_step_diff:
                raise ValueError("Min_step_diff must be less than Max_step_diff")


    def calc_bounds(self):
        '''
        d_bounds: disks' combination number upper bounds
        '''
        self.d_bounds = []

        for max_valid, d_step in zip(self.max_valid_step_diffs, self.d_steps):
            if self.max_step_diff:
                max_valid = min(max_valid, self.max_step_diff)

            # Plus 1 since arange [min, max）
            step_diffs = np.arange(self.min_step_diff, max_valid + 1, 2)
            self.d_bounds.append(np.sum(np.array(d_step) - step_diffs))