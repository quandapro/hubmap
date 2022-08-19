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
                 input_shape = (None, None, 3),
                 encoder_num_heads=[1, 2, 4, 8],
                 encoder_dims=[64, 128, 320, 512],
                 encoder_depth=[2, 2, 2, 2],
                 dropout=0.0,
                 decoder_dim=768,
                 patch_size = [7, 3, 3, 3],
                 stride = [4, 2, 2, 2],
                 sr = [8, 4, 2, 1],
                 activation = 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_num_heads = encoder_num_heads
        self.encoder_dims = encoder_dims
        self.encoder_depth = encoder_depth
        self.dropout=dropout
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.stride = stride
        self.sr = sr
        self.activation = activation

    def mlp(self, inp, dims, dropout_rate, H, W):
        '''
            Multi layer perceptron 
            Dense -> DWConv -> Dense
        '''
        x = inp
        x = Dense(dims * 4)(x)
        x = BatchNormalization()(x)

        B, N, C = tf.shape(x)
        x = tf.reshape(x, (B, H, W, C))
        x = DepthwiseConv2D(kernel_size=3, 
                            strides=(1, 1), 
                            padding='same')(x)
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, (B, H * W, C))
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(dims)(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        return x

    def multi_head_attention(self, query, key, dims, num_heads, dropout_rate):
        B, N_q, _ = tf.shape(query)
        B, N_k, _ = tf.shape(key)

        kv = key

        query = Dense(dims, use_bias=False)(query)
        query = BatchNormalization()(query)
        query = tf.reshape(query, (B, N_q, dims // num_heads, num_heads))

        key   = Dense(dims, use_bias=False)(kv)
        key   = BatchNormalization()(key)
        key   = tf.reshape(key, (B, N_k, dims // num_heads, num_heads))

        value = Dense(dims, use_bias=False)(kv)
        value = BatchNormalization()(value)
        value = tf.reshape(value, (B, N_k, dims // num_heads, num_heads))

        heads = []
        for i in range(num_heads):
            q, k, v = query[..., i], key[..., i], value[..., i]
            attn  = tf.matmul(q, k, transpose_b = True) 
            attn  = BatchNormalization()(attn)
            attn  = K.softmax(attn)

            x     = tf.matmul(attn, v) 
            x     = BatchNormalization()(x)
            heads.append(x)

        # Concatenate heads
        x     = Concatenate(axis=-1)(heads)
        x     = BatchNormalization()(x)
        x     = tf.reshape(x, (B, N_q, dims))

        # Linear projection
        x     = Dense(dims)(x)
        x     = BatchNormalization()(x)
        return x


    def encoder_block(self, inp, patch_size, stride, num_heads, kernels, depth, sr, dropout_rate):
        x = inp
        # Linear convolution embedding
        x = Conv2D(kernels, 
                kernel_size = patch_size,
                padding = 'same',
                strides = stride)(x)
        x = BatchNormalization()(x)

        # Get h, w
        b, h, w, c = tf.shape(x)
        # Flatten x
        x = tf.reshape(x, (b, h * w, c))
        for i in range(depth):
            # Multi Head Attention and skip conn
            shortcuts = x
            x_ = x
            if sr > 1:
                x_ = Conv2D(kernels, 
                            kernel_size = sr,
                            padding = 'same',
                            strides = sr)(tf.reshape(x_, (b, h, w, c)))
                x_ = BatchNormalization()(x_)
                b_, h_, w_, c_ = tf.shape(x_)
                x_ = tf.reshape(x_, (b_, h_ * w_, c_))

            x = self.multi_head_attention(query=x, 
                                          key=x_, 
                                          dims=kernels, 
                                          num_heads=num_heads, 
                                          dropout_rate=dropout_rate)

            x = Add()([shortcuts, x])
            x = BatchNormalization()(x)
            

            # MLP and skip conn
            shortcuts = x
            x = self.mlp(x, kernels, dropout_rate=dropout_rate, H=h, W=w)
            x = Add()([shortcuts, x])
            x = BatchNormalization()(x)

        # Reshape x
        x = tf.reshape(x, (b, h, w, c))
            
        return x

    def decoder(self, inp, dim):
        fuse = []
        _, H, W, _ = tf.shape(inp[0])
        for i in range(len(inp)):
            # Normalize input and reshape
            x = inp[i]
            b, h, w, c = tf.shape(x)
            x = tf.reshape(x, (b, h*w, c))

            # Mlp
            x = Dense(dim)(x)
            x = BatchNormalization()(x)
            x = tf.reshape(x, (b, h, w, dim))

            # UpSampling
            x = tf.image.resize(x, (H, W))

            fuse.append(x)

        # Linear fuse
        fuse = Concatenate(axis=-1)(fuse)
        fuse = BatchNormalization()(fuse)

        x    = Conv2D(dim, kernel_size=1)(fuse)
        x    = BatchNormalization()(x)
        x    = ReLU()(x)
        return x

    def __call__(self):       
        inp = Input(self.input_shape)

        outputs = []

        encoder_blocks = []

        x = BatchNormalization()(inp)

        # Encoder
        for i in range(len(self.encoder_num_heads)):
            x = self.encoder_block(x, 
                                   patch_size=self.patch_size[i],
                                   stride = self.stride[i],
                                   num_heads=self.encoder_num_heads[i], 
                                   kernels=self.encoder_dims[i], 
                                   depth=self.encoder_depth[i], 
                                   sr = self.sr[i],
                                   dropout_rate=self.dropout)
            encoder_blocks.append(x)

        # Decoder
        out = self.decoder(encoder_blocks, self.decoder_dim)
            
        # Final output
        out = Conv2D(self.num_classes, 
                     kernel_size = (1, 1), 
                     padding = 'same')(out)
        out = Activation(self.activation, name=f'output_final')(out)
        outputs.append(out)
        
        # Create model
        model = Model(inputs = inp, outputs = outputs)
        return model