from keras.models import Model,Sequential
from keras.layers import Input,Dense,Flatten, Conv2D,GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from keras.applications.resnet50 import ResNet50


def model_Resnet50(input_shape):
    
    base_Resnet50=ResNet50(include_top = False, weights='imagenet', input_shape=input_shape)
    base_Resnet50.trainable=False
    
    base_Resnet50_output = base_Resnet50.output
    F = Flatten()(base_Resnet50_output)
    FC1 = Dense(100, activation='relu')(F)
    FC2 = Dense(30, activation='relu')(FC1)
    output_layer = Dense(2, activation='sigmoid')(FC2)
    
    model = Model(inputs=base_Resnet50.input, outputs=output_layer)
    
    return model

