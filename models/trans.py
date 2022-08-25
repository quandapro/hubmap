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
                 hidden_dropout=0.0,
                 attention_dropout=0.0,
                 drop_path=0.0,
                 decoder_dim=768,
                 patch_size = [7, 3, 3, 3],
                 stride = [4, 2, 2, 2],
                 sr = [8, 4, 2, 1],
                 deep_supervision = False,
                 activation = 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_num_heads = encoder_num_heads
        self.encoder_dims = encoder_dims
        self.encoder_depth = encoder_depth
        self.hidden_dropout = hidden_dropout
        self.attention_dropout=attention_dropout
        self.drop_path = drop_path
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.stride = stride
        self.sr = sr
        self.activation = activation
        self.deep_supervision = deep_supervision

    def encoder_block(self, inp, patch_size, stride, num_heads, kernels, depth, sr):
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
                                       sr=sr,
                                       dropout_ratio=self.attention_dropout)(x, H, W)
            if self.drop_path > 0:
                x = DropPath(self.drop_path)(x)
            x = Add()([shortcuts, x])
            
            # # MLP and skip conn
            shortcuts = x
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MLP(dims=kernels,
                    mlp_ratio=4,
                    dropout_ratio=self.hidden_dropout)(x, H, W)
            if self.drop_path > 0:
                x = DropPath(self.drop_path)(x)
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
        x    = ConvModule(dim, kernel_size=(1, 1), strides=1, use_bn=True, act=relu)(x)
        return x

    def decoder_custom(self, inp, num_heads, dims, sr):
        decoder_blocks = []

        x = inp[0]
        for i in range(len(inp) - 1):
            # Get query
            q  = inp[i + 1]
            B_q, H, W, C_q = tf.shape(q)
            q = tf.reshape(q, (B_q, -1, C_q))

            # Get Key-value (self) then resize kv to match shape of q
            x = tf.image.resize(x, (H, W))

            # Embedding
            x, H, W = OverlapPatchEmbeddings(1, 1, dims=dims[i + 1])(x)

            # Perform Cross-Attentions where query = encoder features, key/value = segmentation map
            shortcut = x
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MultiHeadCrossAttention(num_heads=num_heads[i + 1], dims = dims[i + 1], sr = sr[i + 1], dropout_ratio=self.attention_dropout)(q, x, H, W)
            x = Add()([shortcut, x])

            # Apply MLP
            shortcut = x
            x = LayerNormalization(epsilon=1e-6)(x) 
            x = MLP(dims[i + 1], mlp_ratio=4, dropout_ratio=self.hidden_dropout)(x, H, W)
            x = Add()([shortcut, x])
            
            # Reshape x
            x = tf.reshape(x, (-1, H, W, dims[i + 1]))

            decoder_blocks.append(x)
        
        return decoder_blocks

    def get_output(self, decoder_blocks):
        decoder_dims = self.encoder_dims[::-1]
        outputs = []
        if self.deep_supervision:
            for i in range(0, len(decoder_blocks) - 1):
                x = LayerNormalization(epsilon=1e-6)(decoder_blocks[i]) 
                x = ConvModule(decoder_dims[i + 1], kernel_size=(1, 1), strides=1, use_bn=True, act=relu)(x)
                x = Conv2D(self.num_classes, 
                            kernel_size = (1, 1), 
                            padding = 'same')(x)
                x = Activation(self.activation, name=f'output_{i}')(x)
                outputs.append(x)

        # Get final output
        x = LayerNormalization(epsilon=1e-6)(decoder_blocks[-1]) 
        x = ConvModule(decoder_dims[-1], kernel_size=(1, 1), strides=1, use_bn=True, act=relu)(x)
        x = Conv2D(self.num_classes, 
                    kernel_size = (1, 1), 
                    padding = 'same')(x)
        x = Activation(self.activation, name=f'output_final')(x)
        outputs.append(x)

        return outputs

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
                                   sr = self.sr[i])
            x = LayerNormalization(epsilon=1e-6)(x)
            encoder_blocks.append(x)

        # Decoder
        decoder = self.decoder(encoder_blocks, self.decoder_dim)

        # decoder = self.decoder_custom(encoder_blocks, self.encoder_num_heads[::-1], self.encoder_dims[::-1], self.sr[::-1])
            
        out = Conv2D(self.num_classes, 
                     kernel_size = (1, 1), 
                     padding = 'same')(decoder)
        out = Activation(self.activation, name=f'output_final')(out)    
        outputs.append(out)
        
        # Create model
        model = Model(inputs = inp, outputs = outputs)
        return model