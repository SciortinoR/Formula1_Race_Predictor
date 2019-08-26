import datetime
import tensorflow as tf
import dataset_formation, data_preprocessing, dnn_arch

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Directories for models & logs
models_dir = "..\Models\DNN_Final-{epoch:02d}-{val_accuracy:.2f}.hdf5"             
logs_dir = "..\Logs\{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Form main dataset
pre_race_df = dataset_formation.create_datasets()

# Preprocess & Generate train, val and test sets
values = data_preprocessing.preprocess_df(pre_race_df)
train_x, train_y, val_x, val_y, test_x, test_y, class_weights = data_preprocessing.generate_dnn_data(values)

# Form/compile deep neural net architecture
dnn_model = dnn_arch.form_dnn(train_x[0].shape)

# Initialize Tensorboard and ModelCheckpoint callbacks 
tensorboard = TensorBoard(log_dir=logs_dir)
checkpoint = ModelCheckpoint(models_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')      # Saves the epoch model if val_accuracy increases from previous epoch

# Train/Validate model
dnn_model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=15,
    class_weight=class_weights,
    validation_data=(val_x, val_y),
    callbacks=[tensorboard, checkpoint])

# Load model once saved (can use currently trained dnn_model as well)
loaded_model = tf.keras.models.load_model('../Models/DNN_Final-14-0.49.hdf5')

# Predict Output/Probability Distributions
predictions = loaded_model.predict(test_x)
print('First prediction: ', predictions[0])

# Test/Score Model
score = loaded_model.evaluate(test_x, test_y, verbose=0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])