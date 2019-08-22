import pandas as pd

def create_datasets():

    # Read seperate datasets
    full_df = pd.read_csv("../Datasets/results.csv",
                          header=None,
                          usecols=[1,2,3,5,6,9,10,17],
                          names=['race_id', 'driver_id', 'constructor_id', 'grid_position', 'final_position', 'race_points_scored', 'race_laps_completed', 'status_id'])
    races = pd.read_csv("../Datasets/races.csv",
                        header=None,
                        usecols=[0,1,2,4],
                        names=['race_id', 'year', 'round', 'race_name'])
    drivers = pd.read_csv("../Datasets/driver.csv", 
                          encoding='latin-1',
                          header=None,
                          usecols=[0,7],
                          names=['driver_id', 'driver_nationality'])
    constructors = pd.read_csv("../Datasets/constructors.csv",
                               header=None,
                               usecols=[0,3],
                               names=['constructor_id', 'constructor_nationality'])
    d_standings = pd.read_csv("../Datasets/driver_standings.csv",
                              header=None,
                              usecols=[1,2,3,4,6],
                              names=['race_id', 'driver_id', 'driver_season_points', 'driver_season_position', 'driver_season_wins'])
    c_standings = pd.read_csv("../Datasets/constructor_standings.csv",
                              header=None,
                              usecols=[1,2,3,4,6],
                              names=['race_id', 'constructor_id', 'constructor_season_points', 'constructor_season_position', 'constructor_season_wins'])
    quali = pd.read_csv("../Datasets/qualifying.csv",
                        header=None,
                        usecols=[1,2,3,6,7,8],
                        names=['race_id', 'driver_id', 'constructor_id', 'q1_time', 'q2_time', 'q3_time'])
    lap_times = pd.read_csv("../Datasets/lap_times.csv",
                            header=None,
                            usecols=[0,1,2,3,5],
                            names=['race_id', 'driver_id', 'lap', 'lap_position', 'lap_time_ms'])
    pit_stops = pd.read_csv("../Datasets/pit_stops.csv",
                            header=None,
                            usecols=[0,1,2,3,6],
                            names=['race_id', 'driver_id', 'race_stop_#', 'race_stop_lap', 'stop_time_ms'])

    # Merging datasets based on column keys
    full_df = pd.merge(full_df, races, how='left')
    full_df = pd.merge(full_df, drivers, how='left')
    full_df = pd.merge(full_df, constructors, how='left')
    full_df = pd.merge(full_df, d_standings, how='left')
    full_df = pd.merge(full_df, c_standings, how='left')
    full_df = pd.merge(full_df, quali, how='left')
    full_df = pd.merge(full_df, lap_times, how='left')
    full_df = pd.merge(full_df, pit_stops, how='left')

    # Slice out in-race data
    pre_race_df = full_df.drop(full_df.columns[[5,6,22,23,24,25,26,27]], axis=1)
    post_race_df = full_df

    # Shuffle for Regular Deep Neural Net
    pre_race_shuffled_df = pre_race_df.sample(frac=1).reset_index(drop=True)
    post_race_shuffled_df = post_race_df.sample(frac=1).reset_index(drop=True)

    # Sort by time for Recurrent Neural Net
    pre_race_df.sort_values(['year', 'round', 'driver_id'], ascending=[True,True, True], inplace=True)
    post_race_df.sort_values(['year', 'round', 'driver_id', 'lap'], ascending=[True, True, True, True], inplace=True)
    pre_race_df.reset_index(drop=True, inplace=True)
    post_race_df.reset_index(drop=True, inplace=True)

    return pre_race_df, pre_race_shuffled_df, post_race_df, post_race_shuffled_df   