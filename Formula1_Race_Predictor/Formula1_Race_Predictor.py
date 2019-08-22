import pandas as pd

# Creating empty dataframes
full_df = pd.DataFrame()
races = pd.DataFrame()
drivers = pd.DataFrame()
constructors = pd.DataFrame()
d_standings = pd.DataFrame()
c_standings = pd.DataFrame()
quali = pd.DataFrame()

# Reading Data
full_df = pd.read_csv("Datasets/results.csv")
races = pd.read_csv("Datasets/races.csv")
drivers = pd.read_csv("Datasets/drivers.csv", encoding='latin-1')
constructors = pd.read_csv("Datasets/constructors.csv")
d_standings = pd.read_csv("Datasets/driverStandings.csv")
c_standings = pd.read_csv("Datasets/constructorStandings.csv")
quali = pd.read_csv("Datasets/qualifying.csv")

# Selecting necessary columns
full_df = full_df[['raceId', 'driverId', 'constructorId', 'grid', 'position']]
races = races[['raceId', 'year', 'round', 'name']]
drivers = drivers[['driverId', 'nationality']]
constructors = constructors[['constructorId', 'nationality']]
d_standings = d_standings[['raceId', 'driverId', 'points', 'position', 'wins']]
c_standings = c_standings[['raceId', 'constructorId', 'points', 'position', 'wins']]
quali = quali[['raceId', 'driverId', 'constructorId', 'q1', 'q2', 'q3']]

# Renaming select columns
full_df.rename({'grid' : 'grid_position', 'position' : 'final_position'}, axis=1, inplace=True)
races.rename({'name' : 'race_name'}, axis=1, inplace=True)
drivers.rename({'nationality' : 'driver_nationality'}, axis=1, inplace=True)
constructors.rename({'nationality' : 'constructor_nationality'}, axis=1, inplace=True)
d_standings.rename({'points' : 'driver_season_points', 'position': 'driver_season_position', 'wins' : 'driver_season_wins'}, axis=1, inplace=True)
c_standings.rename({'points' : 'constructor_season_points', 'position' : 'constructor_season_position', 'wins' : 'constructor_season_wins'}, axis=1, inplace=True)

# Merging datasets based on column keys
full_df = pd.merge(full_df, races, how='left')
full_df = pd.merge(full_df, drivers, how='left')
full_df = pd.merge(full_df, constructors, how='left')
full_df = pd.merge(full_df, d_standings, how='left')
full_df = pd.merge(full_df, c_standings, how='left')
full_df = pd.merge(full_df, quali, how='left')

# Sort by time for Recurrent Neural Net
full_df.sort_values(['year', 'round'], ascending=[True, True], inplace=True)
full_df.reset_index(drop=True, inplace=True)

# Randmoize for Regular Deep Neural Net
shuffled_df = full_df.sample(frac=1).reset_index(drop=True)

print(full_df)
print(shuffled_df)
