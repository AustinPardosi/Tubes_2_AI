import os
import pandas as pd
import pickle
from algorithms.KNN import K_Nearest_Neighbor

# Load Data
train_data = pd.read_csv("data/data_train.csv")
validation_data = pd.read_csv("data/data_validation.csv")

def prepare_data(data):
    points = {}
    for index, row in data.iterrows():
        category = row["price_range"]
        features = row.drop("price_range").tolist()

        if category not in points:
            points[category] = []

        points[category].append(features)

    return points

k = int(input("Masukkan k: "))

# Train model
knn_model = K_Nearest_Neighbor(k)
train_point = prepare_data(train_data)
knn_model.fit(train_point)

# Save the model 
model_folder = "model"
model_filename = os.path.join(model_folder, "knn_model.pkl")
with open(model_filename, "wb") as file:
    pickle.dump(knn_model, file)

# Proses evaluasi model
correct_predictions = 0
total_predictions = len(validation_data)

for index, row in validation_data.iterrows():
    features = row.drop("price_range").tolist()
    true_category = row["price_range"]
    predicted_category = knn_model.predict(features)

    if true_category == predicted_category:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Akurasi data validasi: {accuracy * 100:.2f}%")