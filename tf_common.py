
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf

#define swish activation function
def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish': swish}) 


def set_tensorflow_precision_policy(is_mixed_precision = False):
    '''Set the precision policy of the tensorflow backend to float64 or mixed policy'''
    if is_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        tf.keras.backend.set_floatx('float64')

def get_activation(target_activ):
    '''Return tf.keras activation function
    Args:
        target_activ (str)'''

    activations = {'elu'            : tf.keras.activations.elu,
                    'hard_sigmoid'  : tf.keras.activations.hard_sigmoid,
                    'linear'        : tf.keras.activations.linear,
                    'relu'          : tf.keras.activations.relu,
                    'selu'          : tf.keras.activations.selu,
                    'sigmoid'       : tf.keras.activations.sigmoid,
                    'softmax'       : tf.keras.activations.softmax,
                    'tanh'          : tf.keras.activations.tanh,
                    'swish'         : swish}
    if target_activ not in activations.keys():
            raise Exception('Unsupported activation function specified: {}'.format(target_activ))
    
    activation = activations[target_activ]
    return activation

def get_loss(target_loss):
        '''Return tf.keras loss function'''

        all_losses = {'KLD'         : tf.keras.losses.KLD,
                        'MAE'       : tf.keras.losses.MAE,
                        'MAPE'      : tf.keras.losses.MAPE,
                        'MSE'       : tf.keras.losses.MSE,
                        'mse'       : tf.keras.losses.MSE,
                        'MSLE'      : tf.keras.losses.MSLE,
                        'binary_crossentropy'       : tf.keras.losses.binary_crossentropy,
                        'categorical_crossentropy'  : tf.keras.losses.categorical_crossentropy,
                        'categorical_hinge'         : tf.keras.losses.categorical_hinge,
                        'mse_gs'    : None}

        if target_loss not in all_losses.keys():
            raise Exception('Unsupported loss function specified: {}'.format(target_loss))
        loss = all_losses[target_loss]
        return loss