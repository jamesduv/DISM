#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish': swish})

class keras_nids_v0(tf.keras.Model):
    '''NIDS model. Subclasses tf.keras.Model. and takes arbitrary tf.keras.Models for
    the spatial (main) and parameter (hypernetwork) networks during construction.

    Attributes:
         self.opt (dict) : all model options
         self.spatial_net (tf.keras.Model)  : spatial network
         self.parameter_net (tf.keras.Model): parameter network
         self.call_backs (list)  : container for training callbacks
         self.split_sizes (list) : used for splitting w_hat
         self.wt_mat_shape (tuple) : shape of final layer weight matrix
         Filenames:
            self.fn_csv                 
            self.fn_weights_val_best    
            self.fn_weights_train_best  
            self.fn_weights_end         
            self.fn_model_val_best      
            self.fn_model_train_best    
            self.fn_model_end 

    '''

    def __init__(self, opt, spatial_net, parameter_net):
        super(keras_nids_v0, self).__init__()
        self.opt            = opt
        self.spatial_net    = spatial_net
        self.parameter_net  = parameter_net
        self.call_backs = None
        self.split_sizes    = self.opt['split_sizes']
        self.wt_mat_shape   = self.opt['wt_mat_shape']

    def call(self,x):
        '''Over-ride call(self, x) for use with tf.keras.Model.fit()
        See self.call_model()
        '''

        return(self.call_model(x = x[0], mu = x[1]))

    def call_model(self, x, mu, training=False):
        '''Call the full NIDS model, return only the final result
        
        Args:
            x (ndarray/tensor)     : spatial network inputs, x \in n_pts x n_spatial
            mu (ndarray/tensor)    : parameter network inputs, mu \in n_pts x n_param

        Returns:
            y_pred (ndarray/tensor) : main network output, y_pred \in n_pts x n_state'''

        param_net_out   = self.parameter_net(mu, training=training)
        weight, bias    = tf.split(param_net_out, num_or_size_splits=self.split_sizes, axis=1)
        weight          = tf.reshape(weight, self.wt_mat_shape)
        hx              = self.spatial_net(x, training=training)
        y_pred          = tf.einsum('ijk,ik->ij', weight, hx) + bias
        return y_pred
    
    def call_model_modes(self, x, mu, training=False):
        '''Call the full NIDS model, return final result, hx, and weight/bias
        
        Args:
            x (ndarray/tensor)     : spatial network inputs, x \in n_pts x n_spatial
            mu (ndarray/tensor)    : parameter network inputs, mu \in n_pts x n_param

        Returns:
            y_pred (ndarray/tensor) : main network output
            hx (ndarray/tensor) : final hidden state, hx \in n_pts x n_h
            weight (ndarray/tensor) : output layer weight matrix, weight \in n_pts x n_state x n_h
            bias (ndarray/tensor)   : output layer bias vector, bias \in n_pts x n_h
        '''

        param_net_out   = self.parameter_net(mu, training=training)
        weight, bias    = tf.split(param_net_out, num_or_size_splits=self.split_sizes, axis=1)
        weight          = tf.reshape(weight, self.wt_mat_shape)
        hx              = self.spatial_net(x, training=training)
        y_pred          = tf.einsum('ijk,ik->ij', weight, hx) + bias
        return y_pred, hx, weight, bias

    def set_save_paths(self):
        '''Set paths for saving items during training'''

        self.fn_csv             = os.path.join(self.opt['save_dir'], 'training.csv')

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
        '''Make checkpoints to save the model weights in .h5 format during training'''

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
        '''Make callbacks to save the model in .tf format during training'''
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
            histogram_freq (int) : epoch-frequency to save histograms, use 0 to not save histograms
            profile_batch (int or tuple of int) : batch or batches to profile between
        '''

        self.log_dir = os.path.join(self.opt['save_dir'], 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(log_dir = self.log_dir,
                                                     histogram_freq = histogram_freq,
                                                     write_graph    = True,
                                                     profile_batch  = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)

def get_keras_nids_v0_opt(network_func  = 'keras_nids_v0',
                            name        = '',
                            data_opt    = {},
                            noutput    = 3,
                            nspatial   = 3,
                            nstate     = 3,
                            nparam     = 4,
                            hx_dim      = 50,
                            wt_mat_dim  = '',
                            wt_total_dim = '',
                            split_sizes = [],
                            wt_mat_shape = (),
                            inputs      = ['xc', 'yc', 'sdf'],
                            params      = [],
                            outputs     = [],
                            spatial_opt = None,
                            param_opt   = None,
                            activation  = 'swish',
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
                            kernel_regularizer = None,
                            lambda_l1       = 0,
                            lambda_l2       = 0,
                            save_dir_base   = '../NICE_Predictions',
                            is_debugging    = False):
    if is_learning_rate_decay:
        optimizer_kwargs = {'decay':learning_rate / epochs}
    else:
        optimizer_kwargs = {}

    opt = { 'network_func'  : network_func,
            'name'          : name,
            'data_opt'      : data_opt,
            'noutput'      : noutput,
            'nspatial'     : nspatial,
            'nstate'       : nstate,
            'nparam'       : nparam,
            'hx_dim'        : hx_dim,
            'wt_mat_dim'    : wt_mat_dim,
            'wt_total_dim'  : wt_total_dim,
            'split_sizes'   : split_sizes,
            'wt_mat_shape'  : wt_mat_shape,
            'inputs'        : inputs,
            'params'        : params,
            'outputs'       : outputs,
            'spatial_opt'   : spatial_opt,
            'param_opt'     : param_opt,
            'activation'    : activation,
            'norm_x_by'     : norm_x_by,
            'norm_mu_by'    : norm_mu_by,
            'norm_y_by'     : norm_y_by,
            'batch_size'    : batch_size,
            'learning_rate' :learning_rate,
            'is_learning_rate_decay' : is_learning_rate_decay,
            'training_fraction' :training_fraction,
            'epochs'        : epochs,
            'loss'          : loss,
            'optimizer'     : optimizer,
            'optimizer_kwargs' : optimizer_kwargs,
            'kernel_regularizer' : kernel_regularizer,
            'lambda_l1'     : lambda_l1,
            'lambda_l2'     : lambda_l2,
            'save_dir_base' : save_dir_base,
            'is_debugging'  : is_debugging}
    return opt