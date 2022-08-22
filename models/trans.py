import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.activations import *
from .layers import *
import gc

'''
    SEGFORMER
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

    def encoder_block(self, inp, patch_size, stride, num_heads, kernels, depth, sr, dropout_rate):
        # Linear convolution embedding
        x, H, W = OverlapPatchEmbeddings(patch_size=patch_size,
                                         stride=stride,
                                         dims=kernels)(inp)
        for i in range(depth):
            # Multi Head Attention and skip conn
            shortcuts = x
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MultiHeadSelfAttention(num_heads=num_heads,
                                       dims=kernels,
                                       sr=sr)(x, H, W)
            x = Add()([shortcuts, x])
            
            # MLP and skip conn
            shortcuts = x
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MLP(dims=kernels,
                    mlp_ratio=4,
                    dropout_ratio=dropout_rate)(x, H, W)
            x = Add()([shortcuts, x])

        # Reshape x
        x = tf.reshape(x, (-1, H, W, kernels))
            
        return x

    def decoder(self, inp, dim):
        fuse = []
        _, H, W, _ = tf.shape(inp[0])
        for i in range(len(inp)):
            x = inp[i]
            b, h, w, c = tf.shape(x)
            x = tf.reshape(x, (b, h*w, c))

            # Mlp
            x = Dense(dim)(x)
            x = tf.reshape(x, (b, h, w, dim))

            # UpSampling
            x = tf.image.resize(x, (H, W))

            fuse.append(x)

        # Linear fuse
        x    = Concatenate(axis=-1)(fuse)
        x    = Conv2D(dim,
                      kernel_size = (1, 1), 
                      padding = 'same')(x)
        x    = BatchNormalization(epsilon=1e-6)(x)
        x    = ReLU()(x)
        return x

    def __call__(self):       
        inp = Input(self.input_shape)

        outputs = []

        encoder_blocks = []

        x = inp

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
            x = LayerNormalization(epsilon=1e-6)(x)
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