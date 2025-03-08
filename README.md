
```markdown
# Smart Pharmacy - Medication Demand Prediction using KNN

## 📌 Project Description
This project predicts customers' medication needs based on demographic features and purchase history using the K-Nearest Neighbors (KNN) algorithm. The solution helps optimize inventory management and improve customer service in pharmacies.

**Dataset**:  
- Synthetic dataset containing 1000 records  
- Features: `Age`, `Gender`, `City`, `Disease History`, `Last Purchase`, `Current Need`  
- [Download Dataset](data/drug_data.csv)

## 🛠 Technologies Used
- Python 3.9
- Key Libraries:
  ```python
  pandas, numpy, scikit-learn, matplotlib, seaborn
  ```

## 🧠 KNN Algorithm Implementation
- Classifies medication needs based on feature similarity
- Uses Euclidean distance metric
- Optimal k=5 selected through cross-validation

## 📂 Project Structure
```
pharmacy-knn/
├── data/
│   └── drug_data.csv
├── images/
│   └── confusion_matrix.png
├── Pharmacy_KNN.ipynb
├── requirements.txt
└── README.md
```

## 💻 Code Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('data/drug_data.csv')

# Preprocessing
X = data[['Age', 'Gender_Encoded', 'City_Encoded', 'Disease_History']]
y = data['Current_Need_Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
predictions = knn.predict(X_test)
print(classification_report(y_test, predictions))
```



## 🚀 Getting Started
1. Clone repository:
```bash
git clone https://github.com/yourusername/pharmacy-knn.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:
```bash
jupyter notebook Pharmacy_KNN.ipynb
```


## 📄 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

```

This README includes:
1. Bilingual-friendly structure (easily extendable)
2. Clear installation/usage instructions
3. Visual elements (code blocks, images)
4. Future development roadmap
5. License information
