from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

np.random.seed(42)

data = pd.DataFrame({
    'Speed': np.random.randint(30, 120, 1000),
    'Visibility': np.random.choice([1, 2, 3], size=1000),  # 1: Poor, 2: Moderate, 3: Good
    'Weather': np.random.choice([0, 1], size=1000),  # 0: Clear, 1: Bad
    'Severity': np.random.choice([0, 1, 2], size=1000)  # 0: Minor, 1: Moderate, 2: Severe
})


X = data[['Speed', 'Visibility', 'Weather']]
y = data['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
