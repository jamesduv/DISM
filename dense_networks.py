#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))

tf.keras.utils.get_custom_objects().update({'swish': swish})   

class dense_base():
    '''Base-level functionality for dense networks. 

    Attributes:
        opt (dict)  : all model options
        data (dict) : training data, not required 


    Methods:
        set_kernel_regularizer()
        set_activation()
        set_optimizer()
        set_loss()
    '''

    def __init__(self, opt, data=None):
        self.opt    = opt
        self.data   = data

    def set_kernel_regularizer(self):
        '''Set the kernel regularizer using options defined in self.opt'''

        regularizers = {'none'  : tf.keras.regularizers.l2( l= 0),
                        'l1'    : tf.keras.regularizers.l1( l=self.opt['lambda_l1']),
                        'l2'    : tf.keras.regularizers.l2( l=self.opt['lambda_l2']),
                        'l1_l2' : tf.keras.regularizers.l1_l2( l1=self.opt['lambda_l1'],
                                                            l2=self.opt['lambda_l2']),
                        None    : None}
        if self.opt['kernel_regularizer'] not in regularizers.keys():
                raise Exception('Unsupported kernel regularizer specified: {}'.format(self.opt['kernel_regularizer']))
        self.kernel_regularizer = regularizers[self.opt['kernel_regularizer']]

    def set_activation(self):
        '''Set the activation function using options defined in self.opt'''

        activations = {'elu':tf.keras.activations.elu,
                       'hard_sigmoid':tf.keras.activations.hard_sigmoid,
                       'linear':tf.keras.activations.linear,
                       'relu':tf.keras.activations.relu,
                       'selu':tf.keras.activations.selu,
                       'sigmoid':tf.keras.activations.sigmoid,
                       'softmax':tf.keras.activations.softmax,
                       'tanh':tf.keras.activations.tanh,
                       'swish':swish}
        if self.opt['activation'] not in activations.keys():
                raise Exception('Unsupported activation function specified: {}'.format(self.opt['activation']))
        self.activation = activations[self.opt['activation']]

    def set_optimizer(self):
        '''Set the activation function using options defined in self.opt'''

        optimizers = {'Adadelta':tf.keras.optimizers.Adadelta,
                      'Adagrad':tf.keras.optimizers.Adagrad,
                      'Adam':tf.keras.optimizers.Adam,
                      'Adamax':tf.keras.optimizers.Adamax,
                      'Ftrl':tf.keras.optimizers.Ftrl,
                      'Nadam':tf.keras.optimizers.Nadam,
                      'RMSprop':tf.keras.optimizers.RMSprop,
                      'SGD':tf.keras.optimizers.SGD}
        if self.opt['optimizer'] not in optimizers.keys():
            raise Exception('Unsupported optimizer specified: {}'.format(self.opt['optimizer']))
        self.optimizer = optimizers[self.opt['optimizer']](learning_rate = self.opt['learning_rate'], **self.opt['optimizer_kwargs'])

    def set_loss(self):
        '''Set the loss function using options defined in self.opt'''

        all_losses = {'KLD':tf.keras.losses.KLD,
                  'MAE':tf.keras.losses.MAE,
                  'MAPE':tf.keras.losses.MAPE,
                  'MSE':tf.keras.losses.MSE,
                  'mse':tf.keras.losses.MSE,
                  'MSLE':tf.keras.losses.MSLE,
                  'binary_crossentropy':tf.keras.losses.binary_crossentropy,
                  'categorical_crossentropy':tf.keras.losses.categorical_crossentropy,
                  'categorical_hinge':tf.keras.losses.categorical_hinge}

        if self.opt['loss'] not in all_losses.keys():
            raise Exception('Unsupported loss function specified: {}'.format(self.opt['loss']))
        else:
            self.loss = all_losses[self.opt['loss']]

class dense_v1(dense_base):
    '''Dense network v1 implementation. Does not subclass tf.keras.Model, but
    creates and uses sequential dense tf.keras.Model.
    
    Attributes:
        model (tf.keras.Model)  : dense network 
        call_backs (list)       : container for training callbacks
    
    Methods:
        build_model() : build self.model, using self.opt
        train_model_keras() : train model using model.fit(), w/validation data
        train_model_keras_noval() : train model using model.fit(), no validation data
        set_save_paths() : training utility, make filenames
        make_callbacks_weights() : training utility, callbacks to save weights
        set_training_options()  : set all training options
        start_csv_logger()      : training utility
        tensorboard_callbacks() : training utility, profiling callback
    '''

    def __init__(self, opt, data=None):
        dense_base.__init__(self, opt, data)
        self.set_activation()
        self.call_backs = None

    def build_model(self):
        '''Construct self.model using self.opt'''

        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.opt['input_dim']), name='input')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.opt['n_layers']):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)
            units       = self.opt['n_nodes'][iDense]
            activation  = self.activation

            if iDense == 0:
                output = tf.keras.layers.Dense(units   = units,
                                            activation = activation,
                                            name       = layername,
                                            kernel_initializer  = 'glorot_uniform',
                                            bias_initializer    = 'zeros')(input1)
            else:
                if iDense == (self.opt['n_layers']-1):
                    units = self.opt['n_output']
                    if self.opt['is_linear_output']:
                        activation = tf.keras.activations.linear
                    else:
                        activation = self.activation
                    
                output = tf.keras.layers.Dense(units         = units,
                                                activation   = activation,
                                                name         = layername,
                                                kernel_initializer  = 'glorot_uniform',
                                                bias_initializer    = 'zeros')(output)

        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()

    def train_model_keras(self):
        '''Compile and train the model using self.data, with validation group'''

        self.model.compile(optimizer = self.optimizer, loss=self.loss)
        self.history = self.model.fit(x         = self.data['x_train'],
                                      y         = self.data['y_train'],
                                      epochs    = self.opt['epochs'],
                                      callbacks = self.call_backs,
                                      batch_size = self.opt['batch_size'],
                                      validation_data = (self.data['x_val'], self.data['y_val']))

    def train_model_keras_noval(self):
        '''Compile and train the model using self.data, without validation group'''

        self.model.compile(optimizer = self.optimizer, loss=self.loss)
        self.history = self.model.fit(x         = self.data['x_train'],
                                      y         = self.data['y_train'],
                                      epochs    = self.opt['epochs'],
                                      callbacks = self.call_backs,
                                      batch_size = self.opt['batch_size'])

    def set_save_paths(self):
        '''Set paths for saving items during training'''

        self.fn_csv             = os.path.join(self.opt['save_dir'], 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.opt['save_dir'], 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.opt['save_dir'], 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.opt['save_dir'], 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.opt['save_dir'], 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.opt['save_dir'], 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.opt['save_dir'], 'model.end.tf')

    def make_callbacks_weights(self):
        '''Make checkpoints to save the weights during training. 
        Make checkpoints for:
            1. best validation loss
            2. best training loss
            3. most recent epoch'''

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

    def set_training_options(self):
        '''Set necessary training options'''

        self.set_optimizer()
        self.set_loss()
        self.set_save_paths()
        self.make_callbacks_weights()
        
    def start_csv_logger(self):
        '''Start the csv_logger for training'''

        csv_logger = tf.keras.callbacks.CSVLogger(self.fn_csv)
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(csv_logger)
    
    def tensorboard_callbacks(self, histogram_freq=10, profile_batch=(1,5)):
        '''Make Tensorboard callbacks to profile the model during training
        
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

def set_network_func(opt):
    allowed = {'dense_v1'   : dense_v1}
    netfun  = allowed[opt['network_func']]
    return netfun

def get_dense_opt(network_func  = 'dense_v1',
                  name          = 'dense_test_6',
                  n_layers      = 6,  # number of layers, includes output layer
                  n_nodes       = [50, 50, 50, 50, 50, 1],
                  activation    = 'swish',
                  data_opt      = {},
                  inputs        =  ['xc', 'yc', 'sdf_overall'],
                  input_dim     = 3,
                  n_output      = 1,
                  outputs       = ['q'],
                  norm_x_by     = 'range',
                  norm_y_by     = 'range',
                  is_linear_output = True,
                  is_data_batch     = False,
                  n_data_batches    = 4,
                  batch_size    = 1500,
                  training_fraction = 0.8,
                  epochs        = 5000,
                  loss          = 'mse',
                  optimizer     = 'Adam',
                  n_epochs_save     = 25,
                  learning_rate     = 1e-04,
                  is_learning_rate_decay = False,
                  kernel_regularizer = None,
                  is_load_weights   = False,
                  fn_weights_load   = None,   #weights to use IF continuing
                  is_save_all_weights = True,
                  lambda_l1         = 1e-05,
                  lambda_l2         = 0,
                  save_dir_base     = '../Dense_models/',
                  is_debugging      = False):

    if is_learning_rate_decay:
        optimizer_kwargs = {'decay':learning_rate / epochs}
    else:
        optimizer_kwargs = {}

    opt = {'network_func'   : network_func,
           'name'           : name,
           'n_layers'       : n_layers,
           'n_nodes'        : n_nodes,
           'activation'     : activation,
           'data_opt'       : data_opt,
           'inputs'         : inputs,
           'input_dim'      : input_dim,
           'n_output'       : n_output,
           'outputs'        : outputs,
           'norm_x_by'      : norm_x_by,
           'norm_y_by'      : norm_y_by,
           'is_linear_output' : is_linear_output,
           'is_data_batch'  : is_data_batch,
           'batch_size'     : batch_size,
           'n_data_batches' : n_data_batches,
           'training_fraction'  : training_fraction,
           'epochs'         : epochs,
           'loss'           : loss,
           'optimizer'      : optimizer,
           'optimizer_kwargs' : optimizer_kwargs,
           'n_epochs_save'  : n_epochs_save,
           'learning_rate'  : learning_rate,
           'is_learning_rate_decay' : is_learning_rate_decay,
           'kernel_regularizer' : kernel_regularizer,
           'is_load_weights'    : is_load_weights,
           'fn_weights_load'    : fn_weights_load,
           'is_save_all_weights' : is_save_all_weights,
           'lambda_l1'      : lambda_l1,
           'lambda_l2'      : lambda_l2,
           'save_dir_base'  : save_dir_base,
           'is_debugging'   : is_debugging
           }
    return opt