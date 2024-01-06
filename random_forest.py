from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 23

data = pd.read_csv('power-transformer-maintenance/power_transformer_maintenance_dataset.csv')
X = data[['Age', 'Oil_Quality', 'Temperature', 'Load', 'Humidity', 'Voltage_Fluctuations', 'Frequency_Fluctuations']]
y = data['Maintenance_Need']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=SEED)  # Model

rf_classifier.fit(X_train, y_train)  # Train

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
