# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv("Food_Delivery.csv")
print(df.head())

# Dataset Info
print(df.info())

# Checking null values
print(df.isnull().sum())

# Drop irrelevant columns
df.drop(["Delivery_person_ID", "Restaurant_latitude", "Restaurant_longitude",
         "Delivery_location_latitude", "Delivery_location_longitude"], axis=1, inplace=True)

# Remove rows with NaN values
df.dropna(inplace=True)

# Cleaning data: converting string to float
df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(float)
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')

# Extracting features
df['Day'] = df['Order_Date'].dt.day
df['Month'] = df['Order_Date'].dt.month
df['Year'] = df['Order_Date'].dt.year

# Cleaning 'Time_Orderd' and 'Time_Order_picked'
df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], errors='coerce')
df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce')

# Drop remaining NaN values after conversion
df.dropna(inplace=True)

# Resetting index
df.reset_index(drop=True, inplace=True)

# Convert time to only hour for visualization
df['Hour_Ordered'] = df['Time_Orderd'].dt.hour
df['Hour_Picked'] = df['Time_Order_picked'].dt.hour

# Delivery time as integer
df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(str(x).split(' ')[0]))

# Analysis: Average Delivery Time by Traffic Density
avg_time_traffic = df.groupby("Road_traffic_density")["Time_taken(min)"].mean().reset_index()
print(avg_time_traffic)

# Plot: Average Delivery Time by Traffic Density
sns.barplot(data=avg_time_traffic, x="Road_traffic_density", y="Time_taken(min)")
plt.title("Average Delivery Time by Traffic Density")
plt.ylabel("Avg Time Taken (min)")
plt.xlabel("Traffic Density")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Distribution of Delivery Times
sns.histplot(df["Time_taken(min)"], bins=30, kde=True)
plt.title("Distribution of Delivery Times")
plt.xlabel("Time Taken (min)")
plt.tight_layout()
plt.show()

# Boxplot: Delivery Time by Type of Vehicle
sns.boxplot(data=df, x="Vehicle_condition", y="Time_taken(min)")
plt.title("Delivery Time by Vehicle Condition")
plt.tight_layout()
plt.show()

# Correlation heatmap
numeric_features = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_features.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
