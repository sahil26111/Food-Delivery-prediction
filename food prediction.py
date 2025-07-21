# Installing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Creating a dataframe
df = pd.read_csv("C:/Users/shahi/Downloads/Food_Delivery_Time_Prediction.csv")
df = pd.get_dummies(df, columns=['Weather_Conditions', 'Traffic_Conditions', 'Vehicle_Type'], drop_first=True)
print(df.head())

# Checking for null values
print(df.isnull().sum())

# Standardizing features
scaler = StandardScaler()
scaler.fit_transform(df[['Distance', 'Delivery_Time', 'Order_Cost']])

# Descriptive statistics
descriptive_stats = df.describe()
print(descriptive_stats)

# Boxplot for outlier detection
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[['Distance', 'Delivery_Time', 'Order_Cost']])
plt.title('Boxplot for Outlier Detection')
plt.show()

# Selecting features for binary classification
df_binary = df[['Delivery_Time', 'Distance']]
df_binary.columns = ['time', 'dis']
print(df_binary.head())

# Plotting binary classification
sns.lmplot(x="dis", y="time", data=df_binary, order=2, ci=None)

# Preparing data for linear regression
x = np.array(df_binary['dis']).reshape(-1, 1)
y = np.array(df_binary['time']).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Training linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predicting with the trained model
y_pred = reg.predict(x_test)
print(y_pred)

# Plotting predictions
plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
plt.show()

# Preparing data for pipeline
X = df[['Distance', 'Traffic_Conditions', 'Order_Priority']]
y = df['Delivery_Time']
categorical_features = ['Traffic_Conditions', 'Order_Priority']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R-squared (R²): {r2:.2f}")

# Classification metrics
threshold = 30
y_test_class = (y_test > threshold).astype(int)
predictions_class = (predictions > threshold).astype(int)

accuracy = accuracy_score(y_test_class, predictions_class)
precision = precision_score(y_test_class, predictions_class)
recall = recall_score(y_test_class, predictions_class)
f1 = f1_score(y_test_class, predictions_class)
conf_matrix = confusion_matrix(y_test_class, predictions_class)

print(f"✔️ Accuracy: {accuracy:.2f}")
print(f"🎯 Precision: {precision:.2f}")
print(f"🔁 Recall: {recall:.2f}")
print(f"📏 F1 Score: {f1:.2f}")
print("🧮 Confusion Matrix:")
print(conf_matrix)
