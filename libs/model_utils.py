import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from datetime import datetime, timedelta

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def CONV_stack(X, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack'):    
    bias_flag = not batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        #activation_func = eval(activation)
        
        # linear convolution
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        if activation == 'GELU' or activation == 'gelu':
            X = keras.layers.Activation("gelu", name='{}_{}_activation'.format(name, i))(X)
        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        
    return X
    
def decode_layer(X, channel, pool_size, unpool, kernel_size=3, 
                 activation='ReLU', batch_norm=False, name='decode'):

    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm
    
    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'
    
    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool keyword')
        
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                            padding='same', name='{}_trans_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation == 'GELU' or activation == 'gelu':
            X = keras.layers.Activation("gelu", name='{}_activation'.format(name))(X)
        else:
            if activation is not None:
                activation_func = eval(activation)
                X = activation_func(name='{}_activation'.format(name))(X)
    return X

def encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode'):

    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')
        
    # maxpooling2d as default
    if pool is True:
        pool = 'max'
        
    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm
    
    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)
        
    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size), 
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation == 'GELU' or activation == 'gelu':
            X = keras.layers.Activation("gelu", name='{}_activation'.format(name))(X)
        else:
            if activation is not None:
                activation_func = eval(activation)
                X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

    
    
def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0'):

    pool_size = 2
    
    X = encode_layer(X, channel, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv'.format(name))
    
    return X


def UNET_right(X, X_list, channel, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):
    
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X


def GAN_D(N_envir, N_micro):

    IN_envir = keras.Input((64, 64, N_envir))
    IN_micro = keras.Input((64, 64, N_micro))

    IN = keras.layers.Concatenate(axis=-1)([IN_envir, IN_micro])
    X = IN

    X = keras.layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)

    # pool
    X = keras.layers.Conv2D(64, kernel_size=2, strides=(2, 2), padding='valid', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pool
    X = keras.layers.Conv2D(128, kernel_size=2, strides=(2, 2), padding='valid', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pool
    X = keras.layers.Conv2D(256, kernel_size=2, strides=(2, 2), padding='valid', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    V = X
    V = keras.layers.GlobalMaxPooling2D()(V)
    OUT = keras.layers.Dense(1, activation='sigmoid')(V)

    model_de = keras.Model([IN_envir, IN_micro], OUT)
    
    return model_de

def GAN_G(N_envir, N_micro, filter_input_num=16, filter_num=[32, 64, 128, 256, 512], stack_num_down=2, stack_num_up=2, 
          activation='gelu', batch_norm=True, pool=False, unpool=False, name='unet'):
    
    IN_envir = keras.Input((64, 64, N_envir))
    X1 = IN_envir
    X1 = CONV_stack(X1, filter_input_num, kernel_size=3, stack_num=2, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, 
                       name='{}_input_conv_envir'.format(name))

    IN_micro = keras.Input((64, 64, N_micro))
    X2 = IN_micro
    X2 = CONV_stack(X2, filter_input_num, kernel_size=3, stack_num=2, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, 
                       name='{}_input_conv_micro'.format(name))

    X = keras.layers.Concatenate(axis=-1)([X1, X2])
    X_skip = []
    depth_ = len(filter_num)

    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                      batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)

    # downsampling blocks
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                         batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
        X_skip.append(X)

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        X = UNET_right(X, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                          unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    OUT = keras.layers.Conv2D(N_envir, kernel_size=1)(X)

    model_gen = keras.Model([IN_envir, IN_micro], OUT)
    
    return model_gen

def dummy_loader(model_path):
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W
    


def create_model_vgg(input_shape=(64, 64, 15), channels=[48, 64, 96, 128]):
    
    IN = keras.layers.Input(shape=input_shape)

    X = IN

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)
    
    # pooling
    X = keras.layers.Conv2D(channels[1], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[2], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[3], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    V1 = X
    OUT = keras.layers.GlobalMaxPooling2D()(V1)
    model = keras.Model(inputs=IN, outputs=OUT)
    
    return model


def create_model_head(input_shape=(128,), N_node=64):
    
    IN_vec = keras.Input(input_shape)    
    X = IN_vec
    #
    X = keras.layers.Dense(N_node)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=IN_vec, outputs=OUT)
    
    return model

def create_classif_head(L_vec, k_size=2, padding='same'):
    
    IN = keras.Input((L_vec, 128))
    X = IN
    X = keras.layers.Conv1D(128, kernel_size=k_size, strides=1, padding=padding)(X)
    X = keras.layers.Activation("gelu")(X)
    #
    IN_vec = keras.Input((2,))
    
    X = keras.layers.GlobalMaxPool1D()(X) #X = keras.layers.Flatten()(X)
    X = keras.layers.Concatenate()([X, IN_vec])

    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN, IN_vec], outputs=OUT)
    return model
    
class LayerScale(keras.layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config    
    