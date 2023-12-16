import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("dataset_tables/faculties_tech_uni.xlsx")

# Group the DataFrame by 'Academic Year'
grouped_df = df.groupby('Academic Year')

# Calculate the average number of male students for each academic year
avg_male_students = int(grouped_df['Male Students'].mean())

# Calculate the average number of female students for each academic year
avg_female_students = int(grouped_df['Female Students'].mean())

# Create a bar plot
plt.bar(['Male Students', 'Female Students'], [avg_male_students, avg_female_students], color=['blue', 'yellow'])

# Save the plot
plt.savefig('plots/faculties_tech_uni799.png')

print("The line plot has been saved successfully.")