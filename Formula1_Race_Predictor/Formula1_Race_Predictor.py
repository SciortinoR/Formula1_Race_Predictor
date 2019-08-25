import pandas as pd
import dataset_formation, data_preprocessing

# Form dataset and save it (commented out after saving)
# pre_race_df = dataset_formation.create_datasets()

# Using only pre-race data to make predictions for now
pre_race_df = pd.read_csv("../Datasets/pre_race_df.csv", header=0, index_col=0)

# Remove NaN, Normalize, generate train, val and test sets
values = data_preprocessing.preprocess_df(pre_race_df)
train_x, train_y, val_x, val_y, test_x, test_y = data_preprocessing.generate_dnn_data(values)

# Form deep neural net architecture
# Compile/Train neural net
# Tensorboard validation
# Test with test data
# Save model