# NOT USED
# Transform classes

import numpy as np

class Resize():
	'''
	Rescale image to predetermined size
	'''
	def __init__(self, out_size):
		'''
		out_size: tuple that contains two ints as output img dimension
		'''
		assert isinstance(out_size, tuple)
		self.out_size = out_size

	def __call__(self, img):
		h, w = img.shape[:2]
		out_h, out_w = self.out_size

