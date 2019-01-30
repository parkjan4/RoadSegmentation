import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras.backend as K
from helpers import fmeasure

def create_model1(padding):
    
    model = Sequential()
    nClasses = 2
    
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 16+2*padding, 16+2*padding)
    else:
        input_shape = (16+2*padding, 16+2*padding, 3)
                
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=l2(1e-6)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(nClasses, kernel_regularizer=l2(1e-6)))
     
    return model


def train_network1(train, onehot_tr, padding):
    """Train the network"""
    
    model = create_model(padding)

    # define batch size and number of epochs
    batch_size = 100
    epochs = 200
    
    def softmax_categorical_crossentropy(y_true, y_pred):
        """
        Uses categorical cross-entropy from logits in order to improve numerical stability.
        This is especially useful for TensorFlow (less useful for Theano).
        """
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # compile with a specific loss function and optimization method
    Adam = keras.optimizers.Adam(lr=0.001)
    #model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy', fmeasure])
    model.compile(optimizer=Adam, loss=softmax_categorical_crossentropy, metrics=['accuracy', fmeasure])

    # define path to save optimal weights 
    saving_directory = ""
    
    # slow down when accuracy does not improve any more
    lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    
    # when accuracy converges, stop the training process early
    earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')

    history = model.fit(train, onehot_tr, batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[lr_callback, earlystop])
    
    return model, history

def save_model1(model, path):
    """Save model weights and architecture as json file"""
    model_json = model.to_json()
    with open(path + 'model.json','w') as json_file:
        json_file.write(model_json)
        
    model.save('weights1.hdf5')