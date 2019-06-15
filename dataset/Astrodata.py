# Custom dataset class


import h5py
import numpy as np
from torch.utils.data import Dataset

class Astrodata(Dataset):
    '''
    Astro dataset class
    ''' 

    def __init__(self, data_dir, min_step_diff = None, max_step_diff = None, rtn_log_grid = False, transforms = None, verbose = False):
        '''
        transforms: transformations to apply on imgs
        data_dir: directory to .h5 file
        min_step_diff, max_step_diff: min/max step difference
        rtn_log_grid: return log_grid or not
        verbose: print out debug info
        '''
        # self.data = h5py.File(data_dir, 'r')
        self.data_dir = data_dir
        with h5py.File(self.data_dir, "r") as data:
            self.d_names = list(data.keys())
            # Minus 1 to get rid of "log_grid" in each disk
            self.d_steps = [len(data[d_name].keys()) - 1 for d_name in self.d_names]
        self.min_step_diff = min_step_diff
        self.max_step_diff = max_step_diff
        self.rtn_log_grid = rtn_log_grid
        self.transforms = transforms
        self.verbose = verbose

        self.calc_valid_step_diffs()
        if not self.min_step_diff:
            self.min_step_diff = 2
        if not self.max_step_diff:
            self.max_step_diff = max(self.max_valid_step_diffs)
        self.step_diff_validity_check()
        # self.calc_bounds()
        self.triplets = self.get_map()


    def __getitem__(self, idx):
        '''
        d: disk name
        l, h: Two input imgs for training
        m: label img
        '''
        # d, l_id, h_id, m_id = self.mapping(idx)
        d, l_id, h_id, m_id = self.triplets[idx]
        # print(d, l, h, m)
        with h5py.File(self.data_dir, "r") as data:
            data_d = data[d]
            l = np.array(data_d[str(l_id)])
            h = np.array(data_d[str(h_id)])
            m = np.array(data_d[str(m_id)])
            log_grid = np.asarray(data_d["log_grid"])

        # Apply transforms
        if self.transforms:
            for t_i, transform in enumerate(self.transforms):
                # second transform is a group opration
                # if t_i in self.group_trans_id:
                if hasattr(transform, 'group_tran'):
                    l, h, m = transform(l, h, m)
                    continue
                # Check if func arguments requires of log_grid
                # if 'log_grid' in transform.__call__.__code__.co_varnames:
                if hasattr(transform, 'require_grid'):
                    l = transform(log_grid, l)
                    h = transform(log_grid, h)
                    m = transform(log_grid, m)
                    continue
                l = transform(l)
                h = transform(h)
                m = transform(m)

        ret = []
        if self.rtn_log_grid:
            ret.append(log_grid)
        ret.extend([l, h, m])
        # Verbose mode, return disk name, img indexes
        if self.verbose:
            ret.append({"disk_name": d, 
                        "img1_idx": l_id, 
                        "img2_idx": h_id, 
                        "label_idx": m_id})
        return ret


    def __len__(self):
        # Total number of triplets: 0.25n^2-0.25n-int(0.5n)0.5
        # return sum(self.d_bounds)
        return len(self.triplets)


    def get_map(self):
        '''
        Get List of all possible triplets
        '''
        triplets = []
        for d_id, d_name in enumerate(self.d_names):
            num_dsteps = self.d_steps[d_id]
            max_valid = min(num_dsteps, self.max_step_diff)
            for l in range(num_dsteps - 2):
                h = l + self.min_step_diff
                m = int(0.5 * (h + l))
                while h - l <= max_valid and h < num_dsteps:
                    triplets.append((d_name, l, h, m))
                    h += 2
                    m = l + int(0.5 * (h - l))

        return triplets
        

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

        raise ValueError("Invalid index number {}, maximum index supported is {}".format(idx, self.__len__() - 1))


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
        if self.min_step_diff % 2 != 0:
            raise ValueError("Min_step_diff must be an even number.")

        if self.min_step_diff > max(self.max_valid_step_diffs):
            raise ValueError("Min_step_diff should be less or equal to {}".format(max(self.max_valid_step_diffs)))

        if self.max_step_diff % 2 != 0:
            raise ValueError("Max_step_diff must be an even number.")

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
