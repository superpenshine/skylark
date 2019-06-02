# Custom transform class for pytorch dataloader

import torch
import numpy as np
from PIL import Image

np_pad_name = {'zero':'constant', 
               'circular': 'wrap'}

class CustomPad(object):
    """
    Circular Pad Image
    input:
        pad: int or 4-elements tuple
        img: PIL image
    """

    def __init__(self, pad, mode, **kwargs):
        assert mode in np_pad_name.keys(), \
        'Padding mode zero and circular are implemented for now'
        self.mode = np_pad_name[mode]

        assert isinstance(pad, (int, tuple))
        if isinstance(pad, int):
            self.pad = (pad, pad, pad, pad)
        else:
            assert len(pad) == 4, \
            'Padding length must be an integer or a 4-element tuple'
            self.pad = pad

        self.kwargs = kwargs
        self.numpy_mode = False


    def __call__(self, img):
        '''
        img: ndarray or PIL image
        '''
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError('img should be either PIL Image or ndarray. Got {}'.format(type(img)))
        if isinstance(img, np.ndarray):
            self.numpy_mode = True

        pad_left = self.pad[0]
        pad_top = self.pad[1]
        pad_right = self.pad[2]
        pad_bottom = self.pad[3]

        if not self.numpy_mode and img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), self.mode, **self.kwargs)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        # Turn to RGB if is PIL image
        if not self.numpy_mode:
            img = np.asarray(img)

        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), self.mode, **self.kwargs)

        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), self.mode, **self.kwargs)

        if not self.numpy_mode:
            return Image.fromarray(img)

        return img


class ToTensor(object):
    '''
    Convet numpy array to tensor
    '''
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Custom ToTensor takes numpy array as input")

        return torch.from_numpy(np.transpose(img, (2, 0, 1)))


class GroupRandomCrop(object):
    '''
    Return a random crop at the same position of three input imgs
    input: 2 input images and 1 label image
    '''
    def __init__(self, size, label_size = None):
        '''
        size: height x width
        label_size: label image size
        '''
        assert isinstance(size, (int, tuple)), "Size must be an int or tuple"
        assert isinstance(label_size, (int, tuple)), \
            "Label image size must be an int or tuple"

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2, \
            'Crop output size length must be an integer or a 2-element tuple'
            self.size = size
        if isinstance(label_size, int):
            self.label_size = (label_size, label_size)
        else:
            assert len(label_size) == 2, \
            'Label image output size must be an integer or a 2-element tuple'
            self.label_size = label_size

        self.inner_pad_size = (int(0.5 * (self.size[0] - self.label_size[0])), int(0.5 * (self.size[1] - self.label_size[1])))
        if self.inner_pad_size[0] * self.inner_pad_size[1] < 0:
            raise ValueError(
                "Label size {} must be smaller then img size {}".format(self.label_size, self.size))

        self.numpy_mode = False


    def get_params(self, h, w):
        '''
        Return i, j
        i, j: coordinates of upper left conner
        '''
        i = np.random.randint(0, h - self.size[0])
        j = np.random.randint(0, h - self.size[1])

        return i, j


    def __call__(self, low, high, mid):
        '''
        l, h, m: img_t0, img_t1, img_t0.5
        '''
        if not isinstance(low, (Image.Image, np.ndarray)):
            raise TypeError(
                'img should be either PIL Image or ndarray. Got {}'.format(type(l)))
        if isinstance(low, np.ndarray):
            self.numpy_mode = True

        if self.numpy_mode:
            h, w = low.shape[:2]
        else:
            w, h = low.size

        if self.size[0] > h or self.size[1] > w:
            raise ValueError(
                "Crop output size {}x{} must be <= {}x{}, or pad more first".format(self.size[0], self.size[1], h, w))

        i, j = self.get_params(h, w)
        if not self.numpy_mode:
            return low.crop((j, i, j+self.size[1], i+self.size[0])), \
            high.crop((j, i, j+self.size[1], i+self.size[0])), \
            mid.crop((j+self.inner_pad_size[1], i+self.inner_pad_size[0], j+self.label_size[1], i+self.label_size[0]))
        # print("crop location upper left: {}, {}".format(i, j))
        return low[i:i+self.size[0], j:j+self.size[1]], \
        high[i:i+self.size[0], j:j+self.size[1]], \
        mid[i+self.inner_pad_size[0]:i+self.inner_pad_size[0]+self.label_size[0], j+self.inner_pad_size[1]:j+self.inner_pad_size[1]+self.label_size[1]]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, label_size={})'.format(self.size, self.label_size)


# class GroupRandomCrop2(object):
#     '''
#     Return a random crop at the same position of three input imgs
#     input: 2 input images and 1 label image
#     '''
#     def __init__(self, sizes):
#         '''
#         sizes: ((height, width), (height, width), ...) or (height, width)
#         '''
#         assert isinstance(sizes, (tuple, list)), \
#         "Sizes must be in format" + \
#         "((height, width), (height, width), ...) or (height, width)"

#         self.sizes = []
#         if len(sizes) == 2 and isinstance(sizes[0], int) and isinstance(sizes[1], int):
#             self.sizes.append(list(sizes))
#         else:
#             for pair in sizes:
#                 if not isinstance(pair, (tuple, list)) or \
#                 len(pair) != 2 or \
#                 not isinstance(pair[0], int) or \
#                 not isinstance(pair[1], int):
#                     raise ValueError(
#                         "Sizes must be in format" + 
#                         "((height, width), (height, width), ...) or (height, width)")
#                 self.sizes.append(list(pair))

#         self.numpy_mode = False


    # def get_params(self, h, w):
    #     '''
    #     Return i, j
    #     i, j: coordinates of upper left conner
    #     '''
    #     out_h_max = np.array(self.sizes)
    #     i = np.random.randint(0, h - self.size[0])
    #     j = np.random.randint(0, h - self.size[1])

    #     return i, j


    # def __call__(self, imgs):
    #     '''
    #     l, h, m: img_t0, img_t1, img_t0.5
    #     '''
    #     if not isinstance(low, (Image.Image, np.ndarray)):
    #         raise TypeError(
    #             'img should be either PIL Image or ndarray. Got {}'.format(type(l)))
    #     if isinstance(low, np.ndarray):
    #         self.numpy_mode = True


    #     if self.numpy_mode:
    #         h, w = imgs[0].shape[:2]
    #     else:
    #         w, h = imgs[0].size

    #     for pair in self.sizes:
    #         if self.pair[0] > h or self.pair[1] > w:
    #             raise ValueError(
    #                 "Crop output sizes {} must all <= {}, or pad more first".format(self.sizes, (h, w)))

    #     i, j = self.get_params(h, w)
    #     if not self.numpy_mode:
    #         return low.crop((j, i, j+self.size[1], i+self.size[0])), high.crop((j, i, j+self.size[1], i+self.size[0])), mid.crop((j, i, j+self.size[1], i+self.size[0]))

    #     return low[i:i+self.size[0], j:j+self.size[1]], high[i:i+self.size[0], j:j+self.size[1]], mid[i:i+self.size[0], j:j+self.size[1]]


    # def __repr__(self):
    #     return self.__class__.__name__ + '(size={0})'.format(self.size)