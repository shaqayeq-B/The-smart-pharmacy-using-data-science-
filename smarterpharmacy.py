import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# خواندن داده
data = pd.read_csv('data/drug_data.csv')

# پیشپردازش
X = data[['Age', 'Gender_Encoded', 'City_Encoded', 'Disease_History']]
y = data['Current_Need_Encoded']

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# استانداردسازی
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# آموزش مدل
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ارزیابی
predictions = knn.predict(X_test)
print(classification_report(y_test, predictions))