from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(" Model Saved Successfully")
