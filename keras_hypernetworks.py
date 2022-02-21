#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from dense_networks import dense_base

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish': swish})

class keras_hypernet_v0():
    '''Simple implementation of keras hypernetwork, without using
    tf.keras layers/models for the main network
    
    Attributes:
        self.hypernet (tf.keras model)  : tf.keras hypernetwork, outputs tensor wts = ncases x nwt
        self.opt  (dict)                : all model options
        self.split_sizes (list of int)  : weight dimensions, used to split tensor wts
        self.wt_shapes (list of tuple)  : shape of weight element as tuples, with leading -1, as in (-1, dim1, dim2) to account for batch axis. Axes with -1 are dynamically allocated as required
        '''
        
    def __init__(self, opt, hnet):
        
        self.opt = opt
        # split_size: the dimension of each weight element, 
        # used to split hypernet output
        self.split_sizes    = opt['split_sizes']    
        self.wt_shapes      = opt['wt_shapes']      
        self.hypernet       = hnet
    

    def set_training_fns(self):
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')
        self.fn_weights_best = os.path.join(self.opt['save_dir'], 'weights.best.h5')
        self.fn_weights_end  = os.path.join(self.opt['save_dir'], 'weights.end.h5')

        self.make_weight_save_dir()

    def make_weight_save_dir(self):
        self.weights_dir = os.path.join(self.opt['save_dir'], 'weights_all')
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

    def call_model(self, x, mu, training=False):
        #call hypernet
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
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
    def train_model_ckpt(self, train_data_tup, val_data_tup):
        self.create_training_log_val()
        buffer_size   = 5 * self.opt['batch_size']
        n_data_train    = train_data_tup[0].shape[0]
        n_data_val      = val_data_tup[0].shape[0]
 
        #create dataset for validation data, create batches
        val_data  = tf.data.Dataset.from_tensor_slices(val_data_tup)
        val_data  = val_data.batch(batch_size = self.opt['batch_size'])

        #create the training dataset, shuffle, and create batches
        train_data  = tf.data.Dataset.from_tensor_slices(train_data_tup)

        train_data  = train_data.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)
        train_data  = train_data.batch(batch_size = self.opt['batch_size'])
        n_epochs    = self.opt['epochs']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []

        #save initial weights
        fn_weights_init = os.path.join(self.weights_dir, 'weights.init.h5')
        self.hypernet.save_weights(fn_weights_init)

        for epoch in range(n_epochs):

            #gradient descent over mini-batches
            batch_loss = []
            for batch, (x, mu, q) in enumerate(train_data):
                cur_loss = self.train_on_batch([x, mu], q)
                cur_loss = tf.reduce_sum(cur_loss)
                batch_loss.append(cur_loss)

            # mean_loss = np.mean(np.array(batch_loss))
            total_loss = tf.reduce_sum(batch_loss)
            training_loss.append(total_loss.numpy() / n_data_train)

            #validation loss
            # y_val_pred      = self.call_model([x2_val, x_val])
            # val_loss_cur    = np.mean(self.loss(y_val_pred, y_val))
            # val_loss_store.append(val_loss_cur)

            #validation loss - use batching to avoid OOM
            batch_loss_val = []
            for batch, (x, mu, q) in enumerate(val_data):
                q_val_pred      = self.call_model(x, mu)
                cur_loss_val    = self.loss(q, q_val_pred)
                cur_loss_val    = tf.reduce_sum(cur_loss_val)
                batch_loss_val.append(cur_loss_val)

            # val_loss_cur = tf.reduce_mean(cur_loss_val)
            val_loss_total = tf.reduce_sum(batch_loss_val)
            val_loss_store.append(val_loss_total.numpy() / n_data_val)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                         val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_store[-1]

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.hypernet.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.hypernet.save_weights(self.fn_weights_end)
                message = 'Saving epoch weights'
                print(message)
                self.save_weights_epoch(epoch)
    def train_model(self, train_data_tup, val_data_tup):
        self.create_training_log_val()
        buffer_size   = 5 * self.opt['batch_size']
        n_data_train    = train_data_tup[0].shape[0]
        n_data_val      = val_data_tup[0].shape[0]
 
        #create dataset for validation data, create batches
        val_data  = tf.data.Dataset.from_tensor_slices(val_data_tup)
        val_data  = val_data.batch(batch_size = self.opt['batch_size'])

        #create the training dataset, shuffle, and create batches
        train_data  = tf.data.Dataset.from_tensor_slices(train_data_tup)

        train_data  = train_data.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)
        train_data  = train_data.batch(batch_size = self.opt['batch_size'])
        n_epochs    = self.opt['epochs']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []

        #save initial weights
        fn_weights_init = os.path.join(self.weights_dir, 'weights.init.h5')
        self.hypernet.save_weights(fn_weights_init)

        for epoch in range(n_epochs):

            #gradient descent over mini-batches
            batch_loss = []
            for batch, (x, mu, q) in enumerate(train_data):
                cur_loss = self.train_on_batch([x, mu], q)
                cur_loss = tf.reduce_sum(cur_loss)
                batch_loss.append(cur_loss)

            # mean_loss = np.mean(np.array(batch_loss))
            total_loss = tf.reduce_sum(batch_loss)
            training_loss.append(total_loss.numpy() / n_data_train)

            #validation loss
            # y_val_pred      = self.call_model([x2_val, x_val])
            # val_loss_cur    = np.mean(self.loss(y_val_pred, y_val))
            # val_loss_store.append(val_loss_cur)

            #validation loss - use batching to avoid OOM
            batch_loss_val = []
            for batch, (x, mu, q) in enumerate(val_data):
                q_val_pred      = self.call_model(x, mu)
                cur_loss_val    = self.loss(q, q_val_pred)
                cur_loss_val    = tf.reduce_sum(cur_loss_val)
                batch_loss_val.append(cur_loss_val)

            # val_loss_cur = tf.reduce_mean(cur_loss_val)
            val_loss_total = tf.reduce_sum(batch_loss_val)
            val_loss_store.append(val_loss_total.numpy() / n_data_val)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                         val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_store[-1]

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.hypernet.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.hypernet.save_weights(self.fn_weights_end)
                message = 'Saving epoch weights'
                print(message)
                self.save_weights_epoch(epoch)

    def save_weights_epoch(self, epoch):
        fn_save = os.path.join(self.weights_dir, 'weights.epoch_{:04d}.h5'.format(epoch))
        self.hypernet.save_weights(fn_save)


    @tf.function
    def train_on_batch(self, x, q):
        with tf.GradientTape() as tape:
            q_pred      = self.call_model(x[0], x[1], training=True)
            loss_value  = self.loss(q, q_pred)

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def create_training_log_val(self):
        '''Create training log, with validation data'''
        f           = open(self.fn_train,'a+')
        f.write(','.join(('epoch','loss','val_loss\n')))
        f.close()
    
    def update_training_log_val(self, epoch, training_loss, val_loss):
        '''Append results for single epoch to training log'''
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss)+'\n')))
        f.close()

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        '''Print message to console during training, with validation data'''
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)

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
        self.optimizer = optimizers[self.opt['optimizer']](learning_rate = self.opt['learning_rate'])

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


def hypernet_v0_options(network_func    = 'keras_hypernet_v0',
                        name            = 'poisson_test',
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

    hdims = keras_hypernet_v0_dims(nlayers     = nlayers,
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

def keras_hypernet_v0_dims(nlayers     = 3,
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


class keras_hypernet_v1(tf.keras.Model):
    '''Simple implementation of keras hypernetwork, without using
    tf.keras layers/models for the main network and subclassing tf.keras.Model
    
    Attributes:
        self.hypernet (tf.keras model)  : tf.keras hypernetwork, outputs tensor wts = ncases x nwt
        self.opt (dict)                 : all model options
        self.split_sizes (list of int)  : weight dimensions, used to split tensor wts
        self.wt_shapes (list of tuple)  : shape of weight element as tuples, with leading -1, as in (-1, dim1, dim2) to account for batch axis. Axes with -1 are dynamically allocated as required
        '''
        
    def __init__(self, opt, hypernet):
        super(keras_hypernet_v1, self).__init__()
        self.opt = opt
        self.hypernet = hypernet
        self.call_backs = None

    def call(self,x):
        return(self.call_model(x = x[0], mu = x[1]))
    
    def call_model(self, x, mu, training=False):
        #call hypernet
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
        self.fn_csv             = os.path.join(self.opt['save_dir'], 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.opt['save_dir'], 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.opt['save_dir'], 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.opt['save_dir'], 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.opt['save_dir'], 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.opt['save_dir'], 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.opt['save_dir'], 'model.end.tf')

    def start_csv_logger(self):
        '''Start the csv_logger, OVERWRITES self.call_backs'''
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

        self.log_dir = os.path.join(self.opt['save_dir'], 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(log_dir = self.log_dir,
                                                     histogram_freq = histogram_freq,
                                                     write_graph    = True,
                                                     profile_batch  = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)

    
