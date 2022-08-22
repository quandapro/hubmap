import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
import numpy as np

def get_shape(tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class DropPath(Layer):
    """Stochastic Depth"""
    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob     = 1 - self.drop_path
            x_shape       = get_shape(x)
            shape         = (x_shape[0],) + (1,) * (len(x_shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

class OverlapPatchEmbeddings(Layer):
    """Construct the overlapping patch embeddings."""
    def __init__(self, patch_size, stride, dims, **kwargs):
        super().__init__(**kwargs)
        self.proj = Conv2D(filters=dims, 
                            kernel_size=patch_size, 
                            strides=stride, 
                            padding="same", 
                            name="proj"
        )

        self.layer_norm = LayerNormalization(epsilon=1e-06, name="layer_norm")

    def call(self, x):
        x = self.proj(x)
        B, H, W, C = get_shape(x)
        x = tf.reshape(x, (B, H*W, C))
        x = self.layer_norm(x)
        return x, H, W

class MultiHeadSelfAttention(Layer):
    """Multi Head Self Attention"""
    def __init__(self, num_heads, dims, sr=1, dropout_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dims = dims
        self.query = Dense(dims, name="query") # B x N_q x C 
        self.key   = Dense(dims, name="key")   # B x N_k x C
        self.value = Dense(dims, name="value") # B x N_k x C
        self.proj  = Dense(dims, name="proj")
        self.sr_ratio = sr
        self.dropout = Dropout(dropout_ratio)
        if sr > 1:
            self.sr = Conv2D(filters=dims,
                             kernel_size=sr,
                             strides=sr,
                             padding='same')
            self.layer_norm = LayerNormalization(epsilon=1e-06, name="layer_norm")

    def call(self, x, H, W):
        kv = x
        if self.sr_ratio > 1:
            B, N, C = get_shape(kv)
            kv = tf.reshape(kv, (B, H, W, C))
            kv = self.sr(kv)
            kv = tf.reshape(kv, (B, -1, C))
            kv = self.layer_norm(kv)

        head_dims = self.dims // self.num_heads 
        key_dims  = get_shape(kv)[1]

        query = self.query(x)
        query = tf.reshape(query, (-1, H*W, head_dims, self.num_heads))

        key   = self.key(kv)
        key   = tf.reshape(key, (-1, key_dims, head_dims, self.num_heads))

        value = self.value(kv)
        value = tf.reshape(value, (-1, key_dims, head_dims, self.num_heads))

        heads = []
        for i in range(self.num_heads):
            q, k, v = query[..., i], key[..., i], value[..., i]
            scores  = tf.matmul(q, k, transpose_b=True) / (head_dims ** 0.5)
            weights = tf.nn.softmax(scores, axis=-1)    
            weights = self.dropout(weights)
            
            head    = tf.matmul(weights, v)
            heads.append(head)

        # Concatenate heads
        x = Concatenate(axis=-1)(heads)

        # Combine heads
        x = self.proj(x)
        return x

class MLP(Layer):
    """MLP Layer: Dense -> DWConv -> Dense"""
    def __init__(self, dims, mlp_ratio=4, dropout_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d1     = Dense(dims * mlp_ratio)
        self.dwconv = DepthwiseConv2D(kernel_size=3, 
                                      strides=(1, 1), 
                                      padding='same')
        self.d2     = Dense(dims)
        self.drop   = Dropout(dropout_ratio)

    def call(self, x, H, W):
        x = self.d1(x)

        B, _, C = get_shape(x)
        x = tf.reshape(x, (B, H, W, C))
        x = self.dwconv(x)
        x = gelu(x)
        x = self.drop(x)

        x = tf.reshape(x, (B, H*W, C))
        x = self.d2(x)
        x = self.drop(x)
        return x