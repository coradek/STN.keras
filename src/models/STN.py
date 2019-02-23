from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense

from .utils import get_initial_weights
from .layers import BilinearInterpolation


def STN(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape, name="initial_input")
    locnet = MaxPool2D(pool_size=(2, 2), name="maxp_1")(image)
    locnet = Conv2D(20, (5, 5), name="conv_1")(locnet)
    locnet = MaxPool2D(pool_size=(2, 2), name="maxp_2")(locnet)
    locnet = Conv2D(20, (5, 5), name="conv_2")(locnet)
    locnet = Flatten(name="flatten_1")(locnet)
    locnet = Dense(50, name="dense_1")(locnet)
    locnet = Activation('relu', name="actv_1")(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights, name="dense_2")(locnet)
    x = BilinearInterpolation(sampling_size,
                              name="BilinearInterpolation")([image, locnet])
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Activation('relu')(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # x = Dense(256)(x)
    # x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(inputs=image, outputs=x)
