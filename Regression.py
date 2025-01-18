from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
data = pd.DataFrame({
    'Traffic_Density': np.random.randint(500, 2000, 500),
    'Weather_Condition': np.random.choice([0, 1], size=500),  # 0: Clear, 1: Adverse
    'Road_Condition': np.random.choice([0, 1], size=500),  # 0: Good, 1: Bad
    'Accidents': np.random.randint(0, 50, 500)  # Number of accidents
})


X = data[['Traffic_Density', 'Weather_Condition', 'Road_Condition']]
y = data['Accidents']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
