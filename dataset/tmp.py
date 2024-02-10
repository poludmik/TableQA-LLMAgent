import pandas as pd

df = pd.read_excel('dataset/dataset_tables/private/chargingCyclesHuge.xlsx')

# Task 1: Filter the DataFrame for rows where the 'charger_type' column is equal to 'DC_FAST'
filtered_df = df[df['charger_type (AC/DC/DC_FAST/DC_SUPERCHARGER)'] == 'AC']

# Task 2: Count the number of rows in the filtered DataFrame
frequency_count = filtered_df.shape[0]

# Task 3: Print the count as the frequency of usage of the 'DC_FAST' charger type
print("The frequency of usage of the 'DC_FAST' charger type is:", frequency_count)