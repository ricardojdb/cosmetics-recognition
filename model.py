from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model

def build_mobilenetv2(input_shape=(224,224,3), 
                      hidden_dim=1024, 
                      n_classes=6):

    base_model = MobileNetV2(
        input_shape=input_shape, 
        weights='imagenet', 
        include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(hidden_dim, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, preprocess_input