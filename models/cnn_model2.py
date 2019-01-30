import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras.backend as K
import numpy as np
from helpers import fmeasure

def create_model2(padding):
    
    batch_size = 64 # * 8 for multi-gpu
    num_filters_1 = 16
    num_filters_2 = 32
    num_filters_3 = 64
    num_filters_4 = 128

    # initialize
    model = Sequential()
    nClasses = 2
    
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 16+2*padding, 16+2*padding)
    else:
        input_shape = (16+2*padding, 16+2*padding, 3)
                
    model.add(Conv2D(num_filters_1, kernel_size=(4, 4), input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(num_filters_2, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(num_filters_3, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(num_filters_4, kernel_size=(4,4)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
     
    return model

def train_network2(train, onehot_tr, padding):
    """Train the network"""
    
    model = create_model(padding)

    # define batch size and number of epochs
    batch_size = 100
    epochs = 50
    
    # compile with a specific loss function and optimization method
    Adam = keras.optimizers.Adam(lr=0.001)
    #model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy', fmeasure])
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy', fmeasure])

    # define path to save optimal weights 
    saving_directory = ""
    
    # slow down when accuracy does not improve any more
    lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    
    # when accuracy converges, stop the training process early
    earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')
    
    history = model.fit(train, np.where(onehot_tr==1)[1], batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[lr_callback, earlystop])
    
    return model, history

def save_model2(model, path):
    """Save model weights and architecture as json file"""
    model_json = model.to_json()
    with open(path + 'model.json','w') as json_file:
        json_file.write(model_json)
        
    model.save('weights2.hdf5')