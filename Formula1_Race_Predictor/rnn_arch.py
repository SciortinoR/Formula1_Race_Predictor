import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

def form_rnn(input_shape, num_classes):

    # Define a sequential neural net model
    rnn_model = Sequential()

    # First hidden layer passing inputs to
    rnn_model.add(LSTM(128, input_shape=input_shape, return_sequences=True))        # 128 neurons, LSTM auto uses tanh activation function, return sequences for next rnn lyer
    rnn_model.add(Dropout(0.2))                                                     # Dropout layer in order to regularize and prevent overfitting
    rnn_model.add(BatchNormalization())                                             # Provides a new normalized distribution after passing through the layer

    # Second hidden layer
    rnn_model.add(LSTM(128, return_sequences=True))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(BatchNormalization())

    # Third hidden layer
    rnn_model.add(LSTM(128, return_sequences=True))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(BatchNormalization())

    # Fourth hidden layer
    rnn_model.add(Dense(64, activation='relu'))                                     # Rectified Linear Activation for dense layer
    rnn_model.add(Dropout(0.2))

    # Final output layer
    rnn_model.add(Dense(num_classes, activation='softmax'))                         # Neurons for each final position class, softmax activation 
                                                                                    # gives distributed probability/percentage for each class

    # Optimizer used in compilation/training
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)                            # Learning rate is steps taken in gradient descent
                                                                                    # Decay is decrease in learning rate as get closer to local/global minimum

    # Compile the model
    rnn_model.compile(
        loss='sparse_categorical_crossentropy',                                     # Best for defining loss of a multiclass system 
        optimizer=opt,                                                              # Adam optimizer defined above
        metrics=['accuracy'])                                                       # Track accuracy of system

    return rnn_model
