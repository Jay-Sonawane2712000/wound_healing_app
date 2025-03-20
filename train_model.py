# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Simulate data
num_wounds = 100
num_days = 30
data = []
for wound_id in range(num_wounds):
    wound_type = np.random.choice(['healing', 'infected', 'chronic'])
    if wound_type == 'healing':
        pH = 8.5 - 0.15 * np.arange(num_days) + np.random.normal(0, 0.1, num_days)
    elif wound_type == 'infected':
        pH = 6.5 + 0.1 * np.arange(num_days) + np.random.normal(0, 0.2, num_days)
    else:  # chronic
        pH = 7.5 + 0.05 * np.sin(np.arange(num_days)) + np.random.normal(0, 0.15, num_days)
    
    for day in range(num_days):
        data.append([wound_id, wound_type, day, pH[day]])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Wound_ID', 'Wound_Type', 'Day', 'pH'])
df.to_csv('wound_pH_data.csv', index=False)

# Feature engineering
df['pH_Trend'] = df.groupby('Wound_ID')['pH'].apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
df['pH_Variability'] = df.groupby('Wound_ID')['pH'].transform('std')

# Label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Wound_Type_Encoded'] = encoder.fit_transform(df['Wound_Type'])

# Prepare features and target
features = df.groupby('Wound_ID').agg({'pH': 'mean', 'pH_Trend': 'mean', 'pH_Variability': 'mean'}).reset_index()
target = df.groupby('Wound_ID')['Wound_Type_Encoded'].first().values

# Split data
X_train, X_test, y_train, y_test = train_test_split(features[['pH', 'pH_Trend', 'pH_Variability']], target, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report in the desired format
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("2. Classification Report:")
print(report_df)

# Plot pH trends
plt.figure(figsize=(10, 6))
for wound_type in df['Wound_Type'].unique():
    subset = df[df['Wound_Type'] == wound_type]
    plt.plot(subset['Day'], subset['pH'], label=wound_type)
plt.xlabel('Day')
plt.ylabel('pH')
plt.title('Wound pH Trends')
plt.legend()
plt.grid(True)
plt.annotate('Healing', xy=(15, 6), xytext=(20, 5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)