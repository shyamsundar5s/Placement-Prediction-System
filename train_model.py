import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample data
data = pd.read_csv('placement_data.csv')  # Columns: CGPA, Internships, Projects, Extracurricular, Placed

X = data[['CGPA', 'Internships', 'Projects', 'Extracurricular']]
y = data['Placed']  # 1 = Placed, 0 = Not Placed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
