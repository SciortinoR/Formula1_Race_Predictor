import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import dataset_formation, data_preprocessing, dnn_arch

# Form dataset and save it (commented out after saving)
pre_race_df = dataset_formation.create_datasets()

# Remove NaN, Normalize, generate train, val and test sets
values = data_preprocessing.preprocess_df(pre_race_df)
train_x, train_y, val_x, val_y, test_x, test_y = data_preprocessing.generate_dnn_data(values)

# Form/compile deep neural net architecture
dnn_model = dnn_arch.form_dnn()

# Train/Validate model

# Tensorboard validation
# Test with test data
# Save model