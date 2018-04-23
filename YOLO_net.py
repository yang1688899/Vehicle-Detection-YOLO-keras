from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense

#keras 实现 yolo_v1
# 最终有使用，因为load_weights方法实现有问题
def YOLO_net():
    model = Sequential()
    model.add(Convolution2D(filters=16,kernel_size=(3,3),input_shape=(448,448,3),padding='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=32,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=512,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

    model.add(Convolution2D(filters=1024,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())
    model.add(Dense(1470,activation='linear'))

    return model





