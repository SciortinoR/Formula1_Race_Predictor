import numpy as np
import time
from datetime import datetime
from sklearn import preprocessing

def preprocess_df(df):

    # Iterate over columns removing/replacing NaN values, then normalize/scale
    le = preprocessing.LabelEncoder()
    for col in df.columns:
        if col == 'grid_position':
            df[col].fillna(20, inplace=True)

        # Every non-podium position is considered other (4)
        elif col == 'final_position':
            df[col].fillna(4, inplace=True)
            df[col] = np.where(df[col] > 3, 4, df[col])

        elif col == 'driver_nationality' or col == 'constructor_nationality':
            df[col] = le.fit_transform(df[col])

        elif col == 'constructor_season_points':
            df.dropna(subset=['constructor_season_points', 'constructor_season_position', 'constructor_season_wins'], inplace=True)

        elif col == 'q1_time':
            q1_ms = []
            q2_ms = []
            q3_ms = []

            df[col].fillna('00:00.00', inplace=True)
            df['q2_time'].fillna('00:00.00', inplace=True)
            df['q3_time'].fillna('00:00.00', inplace=True)

            # Converting time stamps into milleseconds
            for q1, q2, q3 in zip(df.q1_time, df.q2_time, df.q3_time):
                q1_obj = datetime.strptime(q1, '%M:%S.%f')
                q2_obj = datetime.strptime(q2, '%M:%S.%f')
                q3_obj = datetime.strptime(q3, '%M:%S.%f')

                q1_ms.append((q1_obj.minute * 60000) + (q1_obj.second * 1000) + (q1_obj.microsecond // 1000))
                q2_ms.append((q2_obj.minute * 60000) + (q2_obj.second * 1000) + (q2_obj.microsecond // 1000))
                q3_ms.append((q3_obj.minute * 60000) + (q3_obj.second * 1000) + (q3_obj.microsecond // 1000))

            df[col] = q1_ms
            df['q2_time'] = q2_ms
            df['q3_time'] = q3_ms

            # Replace all empty quali times with average quali time from that samples grid position
            df[col].replace({0 : np.nan}, inplace=True)
            df['q2_time'].replace({0 : np.nan}, inplace=True)
            df['q3_time'].replace({0 : np.nan}, inplace=True)
            df[col] = df[col].fillna(df.groupby('grid_position')[col].transform('mean'))

            # Replace missing q2 and q3 times with previous quali time
            df['q2_time'].fillna(df['q1_time'], inplace=True)
            df['q3_time'].fillna(df['q2_time'], inplace=True)

        else:
            pass

        # Standardize/scale everything except for target class
        if col != 'final_position':
            df[col] = preprocessing.scale(df[col].values)

    # Final cleanup
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

def generate_rnn_data(df):
    # NOT FINISHED 
    # Sort then split then generate sequences
    df.sort_values(['year', 'round', 'driver_id'], ascending=[True,True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return sequences, y

def generate_dnn_data(df):
    # NOT FINISHED
    # Shuffle then split
    df = df.sample(frac=1).reset_index(drop=True)
    return x, y
