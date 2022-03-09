#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import tf_common

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish': swish})

def keras_hypernet_v1_dims(n_layers_hidden  = 3,
                        n_nodes_hidden      = 10,
                        n_spatial    = 1,
                        n_state      = 1):
    ''''Given the options for the main model, compute the total
    number of weights and biases to generate with the hypernet, and compute
    the size and shape of each element

    Args:
        n_layers_hidden (int)   : number of main network hidden layers
        n_nodes_hidden (int)    : number of nodes per hidden layer
        n_spatial (int)         : number of spatial or main network inputs
        n_state (int)           : number of states or main network outputs 

    Returns:
        weight_info (dict)      : all weight element sizes/shapes
    '''

    #total number of weights and biases in the model to generate
    wt_input_dim  = n_nodes_hidden*n_spatial + n_nodes_hidden
    wt_hidden_dim = n_nodes_hidden**2 + n_nodes_hidden   
    wt_output_dim = n_nodes_hidden*n_state + n_state      

    wt_total_dim = wt_input_dim + ((n_layers_hidden - 1) * (wt_hidden_dim)) + wt_output_dim

    #store shapes of each weight matrix and bias vector
    #leading -1 for batch axis
    wt_shapes = []
    split_sizes = []
    for ii in np.arange(n_layers_hidden+1):
        if ii == 0:
            curshape = (-1, n_nodes_hidden, n_spatial)
        elif ii == n_layers_hidden:
            curshape = (-1, n_state, n_nodes_hidden)
        else:
            curshape = (-1, n_nodes_hidden, n_nodes_hidden)
        wt_shapes.append(curshape)           #weight matrix shape
        wt_shapes.append((-1,curshape[1]))   #bias vector shape

        cursize = curshape[1] * curshape[2]
        split_sizes.append(cursize)
        split_sizes.append(curshape[1])
    n_wt_elem = len(split_sizes)    #number  of weight elements

    weight_info = { 'wt_total_dim'    : wt_total_dim,
                    'wt_shapes'     : wt_shapes,
                    'split_sizes'   : split_sizes,
                    'n_wt_elem'     : n_wt_elem}
    return weight_info

class keras_hypernet_v1(tf.keras.Model):
    '''Design-variable hypernetwork model. Subclasses tf.keras.Model, and takes 
    an arbitrary tf.keras.Model hypernetwork as an argument during construction.

    Use self.call or self.call_model to call the overall model without composing the main 
    network, which evaluates the hypernetwork for every main network input. Used during training.

    Use self.compose_main_network() to generate the weights for a single case (main network), 
    load them into a tf.keras.Model, and return it.
    
    Attributes:
        self.hypernet (tf.keras.Model)  : tf.keras hypernetwork, generates weights/biases
        self.opt (dict)         : all model options, includes entries from keras_hypernet_v1_dims()
        self.call_backs (list)  : container for training callbacks
        self.activation (tf.keras.activation)       : main network activation 
        self.f_activ (list of tf.keras.activation)  : list of main network activations, layerwise
        Filenames:
            self.fn_csv                 
            self.fn_weights_val_best    
            self.fn_weights_train_best  
            self.fn_weights_end         
            self.fn_model_val_best      
            self.fn_model_train_best    
            self.fn_model_end           
            
    Methods:
        call()          : wrapper around self.call_model()  
        call_model()    : forward propagate, without composing main network
        compose_main_network()  : make main network w/generated weights
        build_main_network_model() : make main network w/random weights
        set_save_paths()    : training utility, make filenames
        start_csv_logger()  : training utility
        make_callbacks_weights() : training utility, callbacks to save weights
        make_callbacks_model()   : training utility, callbacks to save model
        tensorboard_callbacks()  : training utility, profiling callback
        build_main_net_activations() : make list of activations, for use with call, call_model
    '''
        
    def __init__(self, opt, hypernet):
        super(keras_hypernet_v1, self).__init__()
        self.opt        = opt
        self.hypernet   = hypernet
        self.call_backs = None
        if self.opt['activation'] is not None:
            self.build_main_net_activations()

    def call(self, x):
        '''Over-ride call(self, x) for use with tf.keras.Model.fit()
        See self.call_model()
        '''

        return(self.call_model(x = x[0], mu = x[1]))
    
    def call_model(self, x, mu, training=False):
        ''' Forward propagate the overall model without composing main network explicitly.
        Evaluate hypernetwork and main network for each input.

        Args:
            x (ndarray/tensor)     : spatial/main network inputs,  x \in n_pts x n_spatial
            mu (ndarray/tensor)    : hypernetwork inputs,          mu \in n_pts x n_param

        Returns:
            output[-1]  (ndarray/tensor) : main network output
        '''

        #generate weights with hypernetwork
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.opt['split_sizes'], axis=1)
        wt_shp = []
        for iShape, target_shape in enumerate(self.opt['wt_shapes']):
            # print(target_shape)
            curwt = tf.reshape(split_weights[iShape], target_shape)
            wt_shp.append(curwt)

        #forward propagate, without composing main model
        output  = []
        output.append(x)
        for iLayer in np.arange(self.opt['n_layers_hidden']+1):
   
            #count by 2's to extract weight, bias from wt_shp
            idx_start   = 2*iLayer
            Wcur        = wt_shp[idx_start]
            bcur        = wt_shp[idx_start + 1]

            #layer multiplication, activation
            zh = tf.einsum('ijk,ik->ij', Wcur, output[iLayer]) + bcur
            hcur = self.f_activ[iLayer](zh)
            output.append(hcur)

        # return weights, split_weights, wt_shp, zh_all, output
        return output[-1]

    def compose_main_network(self, mu, is_print_model_summary = False):
        '''Given a parameter vector mu, call the hypernetwork to generate
        main network weights, and compose the main network using the weights
        in a tf.keras.Model 
        
        Returns:
            model (tf.keras.Model) : main network model, with weights loaded
         '''

        weights         = self.hypernet(mu)
        n_layers_total  = self.opt['n_layers_hidden'] + 1 #hidden + output layer

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.opt['split_sizes'], axis=1)
        wt_shp = []
        wt_squeeze = []
        for iShape, target_shape in enumerate(self.opt['wt_shapes']):
            curwt = tf.reshape(split_weights[iShape], target_shape)
            wt_shp.append(curwt)

            #remove batch axis, transpose
            wt_sq = np.squeeze(curwt, axis=0)
            wt_squeeze.append(wt_sq.T)

        #make list of lists, gathering weights/biases by layer
        #Example: wts_all[0] = (W0, b0)
        wts_all = []
        for iLayer in np.arange(n_layers_total):
            idx1 = iLayer*2
            idx2 = idx1 + 1    
            curlayer_wt = [wt_squeeze[idx1], wt_squeeze[idx2]]
            wts_all.append(curlayer_wt)

        #get tf.keras.Model
        model = self.build_main_network_model()
        if is_print_model_summary:
            model.summary()

        #load the weights into the model
        for iLayer, curwts in enumerate(wts_all):
            curlayer = model.get_layer(index = iLayer+1) #iLayer+1 to skip input layer
            curlayer.set_weights(curwts)

        return model

    def build_main_network_model(self,):
        '''Build and return main network model as specified in self.opt with 
        newly initialized weights, use Keras functional API
        
        Returns:
            model (tf.keras.Model) : main network model, randomly initialized weights
        '''

        n_layers_total  = self.opt['n_layers_hidden'] + 1
        activ = tf_common.get_activation(self.opt['activation'])

        #input layer
        input1 = tf.keras.Input(shape=(self.opt['input_dim']), name='input')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(n_layers_total):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            units       = self.opt['n_nodes_hidden']
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units   = units,
                                            activation = activ,
                                            name       = layername)(input1)
            else:
                #override activ, units for output layer
                if iDense == (n_layers_total - 1):
                    units = self.opt['n_output']
                    if self.opt['is_linear_output']:
                        activ = tf.keras.activations.linear
                    
                output = tf.keras.layers.Dense(units         = units,
                                                activation   = activ,
                                                name         = layername)(output)
        model = tf.keras.Model(inputs=[input1], outputs=output)

        return model

    def set_save_paths(self):
        '''Set paths for saving items during training'''

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
        '''Make checkpoints to save the model weights in .h5 format during training
        
        Make checkpoints for:
            1. best validation loss
            2. best training loss
            3. most recent epoch
        '''

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
        '''Make callbacks to save the model in .tf format during training.
        
        Make checkpoints for:
            1. best validation loss
            2. best training loss
            3. most recent epoch
        '''

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

    def build_main_net_activations(self):
        '''Build list self.f_activ, containing activations for the
        main network as used during training in self.call_model, without composing a separate
        main network tf.keras.Model
        '''

        self.activation = tf_common.get_activation(self.opt['activation'])
        self.f_activ = []
        for iLayer in np.arange(self.opt['n_layers_hidden']+1):
            if iLayer == self.opt['n_layers_hidden']:
                f_activ = tf.keras.activations.linear
            else:
                f_activ = self.activation
            self.f_activ.append(f_activ)

def hypernet_v1_options(network_func    = 'keras_hypernet_v1',
                        name            = None,
                        n_layers_hidden = 5,
                        n_nodes_hidden  = 10,
                        n_spatial        = 3,
                        n_state          = 1,
                        n_param          = 4,
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

    hdims = keras_hypernet_v1_dims(n_layers_hidden     = n_layers_hidden,
                            n_nodes_hidden          = n_nodes_hidden,
                            n_spatial    = n_spatial,
                            n_state      = n_state)
    if is_learning_rate_decay:
        optimizer_kwargs = {'decay':learning_rate / epochs}
    else:
        optimizer_kwargs = {}
    
    opt = {'network_func'   : network_func,
            'name'          : name,
            'n_layers_hidden'   : n_layers_hidden,
            'n_nodes_hidden'    : n_nodes_hidden,
            'n_spatial'     : n_spatial,
            'input_dim'     : n_spatial,
            'n_state'        : n_state,
            'n_output'       : n_state,
            'n_param'        : n_param,
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