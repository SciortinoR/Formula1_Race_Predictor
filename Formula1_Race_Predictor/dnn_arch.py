import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def form_dnn(input_shape, num_classes):
    
    # Define a sequential neural net model
    dnn_model = Sequential()

    # First hidden layer passing inputs to
    dnn_model.add(Dense(128, input_shape=input_shape, activation='relu'))       # 128 neurons, using rectified linear activation function
    dnn_model.add(Dropout(0.2))                                                 # Dropout layer in order to regularize or prevent overfitting
    dnn_model.add(BatchNormalization())                                         # Provides a new normalized distribution after passing through the layer

    # Second hidden layer
    dnn_model.add(Dense(128, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(BatchNormalization())

    # Third hidden layer
    dnn_model.add(Dense(128, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(BatchNormalization())

    # Final output layer
    dnn_model.add(Dense(num_classes, activation='softmax'))                     # Neurons for each final position class, softmax activation 
                                                                                # gives distributed probability/percentage for each class

    # Optimizer used in compilation/training
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)                        # Learning rate is steps taken in gradient descent
                                                                                # Decay is decrease in learning rate as get closer to local/global minimum

    # Compile the model
    dnn_model.compile(
        loss='sparse_categorical_crossentropy',                                 # Best for defining loss of a multiclass system 
        optimizer=opt,                                                          # Adam optimizer defined above
        metrics=['accuracy'])                                                   # Track accuracy of system

    return dnn_model