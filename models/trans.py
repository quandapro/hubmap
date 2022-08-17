from ctypes import resize
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.activations import *
import gc
'''
    BASELINE 2D-UNET WITH DEEP SUPERVISION
'''
class TranSeg:
    def __init__(self, num_classes = 1, 
                 input_shape = (None, None, 1),
                 encoder_num_heads=[1, 1, 1, 1],
                 encoder_dims=[16, 32, 64, 128],
                 encoder_depth=[1, 1, 1, 1],
                 dropout=0.0,
                 decoder_dims =[128, 64, 32, 16],
                 sr = [8, 4, 2, 1],
                 deep_supervision=True,
                 activation = 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_num_heads = encoder_num_heads
        self.encoder_dims = encoder_dims
        self.encoder_depth = encoder_depth
        self.dropout=dropout
        self.decoder_dims = decoder_dims
        self.sr = sr
        self.deep_supervision = deep_supervision
        self.activation = activation

    def encoder_block(self, inp, num_heads, kernels, depth, dropout_rate, stride):
        x = inp
        # Linear convolution embedding
        x = Conv2D(kernels, 
                   kernel_size = stride,
                   padding = 'same',
                   strides = stride)(x)
        x = BatchNormalization()(x)

        for i in range(depth):
            ## Multi-head attention
            shortcuts = x

            x = MultiHeadAttention(num_heads = num_heads, key_dim = kernels, dropout=dropout_rate)(x, x)

            # Skip connection
            x = Add()([shortcuts, x])
            x = BatchNormalization()(x)
            x = gelu(x)

            ## Apply mlp
            shortcuts = x

            # Mlp
            x = Dense(kernels * 4)(x)
            x = gelu(x)
            x = Dense(kernels)(x)
            x = gelu(x)

            # Skip connection
            x = Add()([shortcuts, x])
            x = BatchNormalization()(x)
            x = gelu(x)

        return x


    def decoder_block(self, inp, dims, resize_shape, shortcuts=None):
        x = inp
        x = tf.image.resize(x, resize_shape)
        if shortcuts is not None:
            x = Concatenate()([x, shortcuts])

        # Apply mlp
        x = Dense(dims * 4)(x)
        x = gelu(x)
        x = Dense(dims)(x)
        x = gelu(x)

        return x

    def __call__(self):       
        inp = Input(self.input_shape)

        outputs = []

        encoder_blocks = []

        x = inp

        # Encoder
        for i in range(len(self.encoder_num_heads)):
            x = self.encoder_block(x, self.encoder_num_heads[i], self.encoder_dims[i], self.encoder_depth[i], dropout_rate=self.dropout, stride=self.sr[i] )
            encoder_blocks.append(x)

        out = encoder_blocks[-1]

        # Decoder
        for i in range(len(self.decoder_dims)):
            shortcut = None
            if i < len(encoder_blocks) - 1:
                shortcut = encoder_blocks[len(encoder_blocks) - i - 2]
            if shortcut is not None:
                out = self.decoder_block(out, self.decoder_dims[i], tf.shape(shortcut)[1:3], shortcut)
            else:
                out = self.decoder_block(out, self.decoder_dims[i], tf.shape(out)[1:3], shortcut)

            # Get deep supervision output
            if self.deep_supervision and len(self.decoder_dims) - 4 < i < len(self.decoder_dims) - 1:
                pred = Conv2D(self.num_classes, 
                              kernel_size = (1, 1), 
                              padding = 'same')(out)
                pred = tf.image.resize(pred, tf.shape(inp)[1:3])
                pred = Activation(self.activation, name = f'output_{i}')(pred)
                outputs.append(pred)
            
        # Final output
        out = Conv2D(self.num_classes, 
                     kernel_size = (1, 1), 
                     padding = 'same')(out)
        out = tf.image.resize(out, tf.shape(inp)[1:3])
        out = Activation(self.activation, name=f'output_final')(out)
        outputs.append(out)
        
        # Create model
        model = Model(inputs = inp, outputs = outputs)
        return model