#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from dense_networks import dense_base

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish': swish})

def keras_hypernet_v1_dims(nlayers     = 3,
                        nh          = 10,
                        nspatial    = 1,
                        nstate      = 1):
    ''''Given the options for the main model, compute the total
    number of weights required from the hypernet, and compute
    the sizes and shapes of each weight matrix/vector'''
    
    #total number of weights in the model to predict
    wt_input_dim  = nh*nspatial + nh     #first hidden layer # weights
    wt_hidden_dim = nh**2 + nh          #other hidden layer # weights
    wt_output_dim = nh*nstate + nstate  #output layer # weights
    wt_total_dim  =  ((nh*nspatial) + nh) + ((nlayers-1)*nh*(nh+1)) + (nstate*(nh+1))

    #store shapes of each weight matrix and bias vector
    #leading -1 for batch axis
    wt_shapes = []
    split_sizes = []
    for ii in np.arange(nlayers+1):
        if ii == 0:
            curshape = (-1, nh, nspatial)
        elif ii == nlayers:
            curshape = (-1, nstate, nh)
        else:
            curshape = (-1, nh, nh)
        wt_shapes.append(curshape)           #weight matrix shape
        wt_shapes.append((-1,curshape[1]))   #bias vector shape

        cursize = curshape[1] * curshape[2]
        split_sizes.append(cursize)
        split_sizes.append(curshape[1])
    n_wt_elem = len(split_sizes)    #number  of weight elements

    retval = {'wt_total_dim'    : wt_total_dim,
                'wt_shapes'     : wt_shapes,
                'split_sizes'   : split_sizes,
                'n_wt_elem'     : n_wt_elem}
    return retval

## TODO: update call_model, assign activations ahead of time, allow for activation other than swish
class keras_hypernet_v1(tf.keras.Model):
    '''Simple implementation of keras hypernetwork, without using
    tf.keras layers/models for the main network and subclassing tf.keras.Model
    
    Attributes:
        self.hypernet (tf.keras model)  : tf.keras hypernetwork, outputs tensor wts = ncases x nwt
        self.opt (dict)                 : all model options
            self.opt['split_sizes'] (list of int)  : weight/bias dimensions, used to split generated weight tensor
            self.opt['wt_shapes'] (list of tuple)  : shape of weight element as tuples, with leading -1, as in (-1, dim1, dim2) to account for batch axis. Axes with -1 are dynamically allocated as required
        '''
        
    def __init__(self, opt, hypernet):
        super(keras_hypernet_v1, self).__init__()
        self.opt        = opt
        self.hypernet   = hypernet
        self.call_backs = None

    def call(self, x):
        return(self.call_model(x = x[0], mu = x[1]))
    
    def call_model(self, x, mu, training=False):

        #generate weights with hypernetwor
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.opt['split_sizes'], axis=1)
        wt_shp = []
        for iShape, target_shape in enumerate(self.opt['wt_shapes']):
            # print(target_shape)
            curwt = tf.reshape(split_weights[iShape], target_shape)
            wt_shp.append(curwt)

        #forward propagate
        output  = []
        output.append(x)
        for iLayer in np.arange(self.opt['nlayers']+1):

            #set activation
            if iLayer == self.opt['nlayers']:
                f_activ = tf.keras.activations.linear
            else:
                f_activ = swish
            
            #count by 2's to extract weight, bias from wt_shp
            idx_start   = 2*iLayer
            Wcur        = wt_shp[idx_start]
            bcur        = wt_shp[idx_start + 1]

            #layer multiplication
            zh = tf.einsum('ijk,ik->ij', Wcur, output[iLayer]) + bcur
            hcur = f_activ(zh)
            output.append(hcur)

        # return weights, split_weights, wt_shp, zh_all, output
        return output[-1]

    def set_save_paths(self):
        self.fn_csv                 = os.path.join(self.opt['save_dir'], 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.opt['save_dir'], 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.opt['save_dir'], 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.opt['save_dir'], 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.opt['save_dir'], 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.opt['save_dir'], 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.opt['save_dir'], 'model.end.tf')

    def start_csv_logger(self):
        '''Start the csv_logger for training'''
        csv_logger = tf.keras.callbacks.CSVLogger(self.fn_csv)
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(csv_logger)
          
    def make_callbacks_weights(self):
        '''Make checkpoints to save the weights only during training'''

        if self.call_backs is None:
            self.call_backs = []

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_val_best, 
                                     monitor = 'val_loss',
                                     verbose = 1,
                                     save_best_only = True,
                                     mode = 'min',
                                     save_weights_only=True)
        self.call_backs.append(checkpoint)
        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_end,
                                                        monitor='val_loss',
                                                        verbose=1, 
                                                        save_best_only=False,
                                                        save_weights_only=True)
        self.call_backs.append(checkpoint_2)

        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_train_best, 
                                     monitor = 'loss',
                                     verbose = 1,
                                     save_best_only = True,
                                     mode = 'min',
                                     save_weights_only=True)
        self.call_backs.append(checkpoint_3)

    def make_callbacks_model(self):
        '''Make callbacks to save the model to file'''

        if self.call_backs is None:
            self.call_backs = []
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_model_val_best, 
                                     monitor = 'val_loss',
                                     verbose = 1,
                                     save_best_only = True,
                                     mode = 'min',
                                     save_weights_only=False,
                                     save_format='.tf')
        self.call_backs.append(checkpoint)

        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(self.fn_model_end,
                                                          monitor='val_loss',
                                       verbose=1, save_best_only=False,
                                       save_weights_only=False,
                                       save_format='.tf')
        self.call_backs.append(checkpoint_2)

        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(self.fn_model_train_best, 
                                     monitor = 'loss',
                                     verbose = 1,
                                     save_best_only = True,
                                     mode = 'min',
                                     save_weights_only=False,
                                     save_format='.tf')
        self.call_backs.append(checkpoint_3)

    def tensorboard_callbacks(self, histogram_freq=10, profile_batch=(1,5)):
        '''Create tensorboard callbacks
        Args:
            histogram_freq (int)                : epoch frequency to save histograms, use 0 to not save histograms
            profile_batch (int or tuple of int) : batch or batches to profile between'''

        self.log_dir = os.path.join(self.opt['save_dir'], 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(log_dir = self.log_dir,
                                                     histogram_freq = histogram_freq,
                                                     write_graph    = True,
                                                     profile_batch  = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)

def hypernet_v1_options(network_func    = 'keras_hypernet_v1',
                        name            = None,
                        nlayers         = 5,
                        nh              = 10,
                        nspatial        = 3,
                        nstate          = 1,
                        nparam          = 4,
                        activation      = 'swish',
                        is_linear_output = True,
                        data_opt        = {},
                        inputs          = ['xc', 'yc', 'sdf_overall'],
                        params          = ['xcen', 'ycen', 'radius', 'rotation'],
                        outputs         = ['q'],
                        norm_x_by       = 'range',
                        norm_mu_by      = 'range',
                        norm_y_by       = 'range',
                        batch_size      = 1500,
                        learning_rate   = 1e-04,
                        is_learning_rate_decay = True,
                        training_fraction = 0.8,
                        epochs          = 2000,
                        loss            = 'mse',
                        optimizer       = 'Adam',
                        n_epochs_save   = 10,
                        kernel_regularizer = None,
                        lambda_l1       = 0,
                        lambda_l2       = 0,
                        save_dir_base   = '../Hypernet_models',
                        is_debugging    = True,
                        is_linear_hypernetwork = False,
                        hypernetwork_type = 'dense_v1'):

    hdims = keras_hypernet_v1_dims(nlayers     = nlayers,
                            nh          = nh,
                            nspatial    = nspatial,
                            nstate      = nstate)
    if is_learning_rate_decay:
        optimizer_kwargs = {'decay':learning_rate / epochs}
    else:
        optimizer_kwargs = {}
    
    opt = {'network_func'   : network_func,
            'name'          : name,
            'nlayers'       : nlayers,
            'nh'            : nh,
            'nspatial'      : nspatial,
            'input_dim'     : nspatial,
            'nstate'        : nstate,
            'noutput'       : nstate,
            'nparam'        : nparam,
            'activation'    : activation,
            'is_linear_output' : is_linear_output,
            'data_opt'      : data_opt,
            'inputs'        : inputs,
            'outputs'       : outputs,
            'params'        : params,
            'norm_x_by'     : norm_x_by,
            'norm_mu_by'    : norm_mu_by,
            'norm_y_by'     : norm_y_by,
            'batch_size'    : batch_size,
            'learning_rate' : learning_rate, 
            'is_learning_rate_decay' : is_learning_rate_decay,
            'training_fraction' : training_fraction,
            'epochs'        : epochs,
            'loss'          : loss,
            'optimizer'     : optimizer,
            'n_epochs_save' : n_epochs_save, 
            'kernel_regularizer' : kernel_regularizer, 
            'lambda_l1'     : lambda_l1,
            'lambda-l2'     : lambda_l2,
            'save_dir_base' : save_dir_base,
            'is_debugging'  : is_debugging,
            'is_linear_hypernetwork' : is_linear_hypernetwork,
            'hypernetwork_type' : hypernetwork_type,
            **hdims}
    return opt