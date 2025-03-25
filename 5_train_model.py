from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

X_train = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\X_train.npy")
y_train = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\y_train.npy")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "final_voice_classifier.pkl")

print("✅ Model o‘qitildi va saqlandi!")
