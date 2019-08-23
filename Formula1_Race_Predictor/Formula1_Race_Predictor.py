import dataset_formation, data_preprocessing

# Using only pre-race data to make predictions for now
pre_race_df = dataset_formation.create_datasets()

# Remove NaN & Normalize
data_preprocessing.preprocess_df(pre_race_df)
print(pre_race_df)


# Preprocess data
# Build neural net architectures

# Add class_weights for balancing before fitting
# Train neural nets 
# Tensorboard validation
# Race Prediction

# Maybe have this file only load the models and predict, have train and save in another file.**