from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

X_test = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\X_test.npy")
y_test = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\y_test.npy")

model = joblib.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\final_voice_classifier.pkl")

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
