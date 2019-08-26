import datetime
import numpy as np
import tensorflow as tf
import dataset_formation, data_preprocessing, dnn_arch, rnn_arch

from numpy import concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Directories for models & logs
dnn_models_dir = "..\Models\DNN\E_{epoch:02d}--VA_{val_accuracy:.2f}.hdf5"
rnn_models_dir = "..\Models\RNN\E_{epoch:02d}--VA_{val_accuracy:.2f}.hdf5"          
dnn_logs_dir = "..\Logs\DNN\{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
rnn_logs_dir = "..\Logs\RNN\{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Form main dataset
pre_race_df = dataset_formation.create_datasets()

# Preprocess & Generate train, val and test sets
values, class_weights = data_preprocessing.preprocess_df(pre_race_df)
#train_x, train_y, val_x, val_y = data_preprocessing.generate_rnn_data(values)
train_x, train_y, val_x, val_y, test_x, test_y = data_preprocessing.generate_dnn_data(values)

# Form/compile deep neural net architecture
#rnn_model = rnn_arch.form_rnn(train_x[0].shape, len(np.unique(train_y)))
dnn_model = dnn_arch.form_dnn(train_x[0].shape, len(np.unique(train_y)))

# Initialize Tensorboard and ModelCheckpoint callbacks 
tensorboard = TensorBoard(log_dir=dnn_logs_dir)
checkpoint = ModelCheckpoint(dnn_models_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')      # Saves the epoch model if val_accuracy increases from previous epoch

# Train/Validate model
dnn_model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=100,
    class_weight=class_weights,
    validation_data=(val_x, val_y),
    callbacks=[tensorboard, checkpoint])

# Load model once saved (can use currently trained dnn_model as well)
#loaded_model = tf.keras.models.load_model('../Models/DNN/E_15--VA_0.50.hdf5')

# DNN Predicitions/Scoring
# (Reverse normalization on sample to see which sample predicting)
predictions = dnn_model.predict(test_x)
ty_reshaped = test_y.reshape(test_y.shape[0], 1)
tx_conc = concatenate((test_x, ty_reshaped), axis=1)
tx_conc = data_preprocessing.scaler.inverse_transform(tx_conc)
print('Sample Data: ', tx_conc[0])
print('Sample Probability Distribution: ', predictions[0])

# Test/Score DNN Model
score = dnn_model.evaluate(test_x, test_y, verbose=0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

## RNN predictions/scoring
## (Reverse normalization on sample to see which sample predicting)
#predictions = rnn_model.predict(val_x)
#ty_reshaped = val_y.reshape(val_y.shape[0], 1)
#tx_conc = val_x.reshape(val_x.shape[0], val_x.shape[2])
#tx_conc = concatenate((tx_conc, ty_reshaped), axis=1)
#tx_conc = data_preprocessing.scaler.inverse_transform(tx_conc)
#print('Sample Data: ', tx_conc[0])
#print('Sample Probability Distribution: ', predictions[0])

## Test/Score RNN Model
#score = rnn_model.evaluate(val_x, val_y, verbose=0)
#print('Test Loss: ', score[0])
#print('Test Accuracy: ', score[1])