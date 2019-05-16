# Custom Padding class

from torch.nn.functional import _pad_circular
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.modules.utils import _quadruple
from torch.nn.modules.module import Module

@weak_module
class _CircularPadNd(Module):
    __constants__ = ['padding']

    @weak_script_method
    def forward(self, input):
        return F.pad(input, self.padding, 'circular')

    def extra_repr(self):
        return '{}'.format(self.padding)

@weak_module
class CircularPad2d(_CircularPadNd):
    '''
    Circular Padding
    '''
    def __init__(self, padding):
        super(CircularPad2d, self).__init__()
        self.padding = _quadruple(padding)
