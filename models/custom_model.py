import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K

import segmentation_models as sm

class CustomModel:
    def __init__(self, model_name, input_shape, classes, encoder_weights, activation, encoder_features):
        self.model_name = model_name
        self.input_shape = input_shape
        self.classes = classes
        self.encoder_weights = encoder_weights
        self.activation = activation
        self.encoder_features = encoder_features

    def __call__(self):
        sm.set_framework("tf.keras")
        sm_model = sm.Unet(self.model_name, input_shape=self.input_shape, classes=self.classes, encoder_weights=self.encoder_weights, activation=self.activation)
        decoder_output = ["decoder_stage3b_relu", "decoder_stage2b_relu"]
        outputs = []
        for i, x in enumerate(decoder_output):
            pool_size = 2**(i + 1)
            x = sm_model.get_layer(x).output
            pred = Conv2D(self.classes, 
                        kernel_size = (1, 1), 
                        padding = 'same')(x)
            pred = UpSampling2D(pool_size)(pred)
            pred = Activation(self.activation, name = f'output_{i}')(pred)
            outputs.append(pred)
        outputs += sm_model.outputs
        return Model(inputs = sm_model.inputs, outputs = outputs)

