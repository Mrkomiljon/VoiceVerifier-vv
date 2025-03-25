import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\final_embeddings.npy")
y = np.load("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\final_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\X_train.npy", X_train)
np.save("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\X_test.npy", X_test)
np.save("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\y_train.npy", y_train)
np.save("C:\\Users\\GOOD\\Desktop\\Komil\\Call_center\\embedding_labels\\y_test.npy", y_test)

print("✅ Dataset train va testga bo‘lindi!")
