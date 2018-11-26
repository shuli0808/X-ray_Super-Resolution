import tensorflow as tf
from tensorflow.keras import layers

class Srcnn(tf.keras.Model):
    def __init__(self, image_size=33, label_size=21, c_dim=1):
        super(Srcnn, self).__init__(name='srcnn')
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.is_grayscale = (c_dim == 1)

        # Define your layers here.
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=(9, 9),
                                     strides=(1,1), padding='valid', 
                                     data_format="channels_last", 
                                     activation='relu', use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros')
        self.conv_2 = layers.Conv2D(filters=32, kernel_size=(1, 1),
                                     strides=(1,1), padding='valid', 
                                     data_format="channels_last", 
                                     activation='relu', use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros')
        self.conv_3 = layers.Conv2D(filters=c_dim, kernel_size=(5, 5),
                                     strides=(1,1), padding='valid', 
                                     data_format="channels_last", 
                                     activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros',)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-3:-1] = self.label_size 
        return tf.TensorShape(shape)
