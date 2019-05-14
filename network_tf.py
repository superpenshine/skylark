# Network class

import pdb
import numpy as np
import tensorflow as tf


class nn():
    '''
    Network Class
    '''

    def __init__(self, config):
        
        self.config = config

        self._init_tensorflow()
        self._build_writer()


    def _init_tensorflow(self):
        '''
        Init session
        '''
        self.sess = tf.Session()


    def load_img_on_tb(self, data, grid):
        '''
        Display on tensorboard
        '''
        # Reshape data
        img = data[:,:,1] # Display dust on img
        img = np.reshape(img, (-1, img.shape[0], data.shape[1], 1))
        # pdb.set_trace()
        summary_op = tf.summary.image("test_img0", img, max_outputs=1)
        summary = self.sess.run(summary_op)
        self.writer.add_summary(summary)
        self.writer.close()


    def _build_writer(self):
        '''
        Build the writers
        '''
        log_dir = self.config.log_dir / "test"
        self.writer = tf.summary.FileWriter(log_dir)



