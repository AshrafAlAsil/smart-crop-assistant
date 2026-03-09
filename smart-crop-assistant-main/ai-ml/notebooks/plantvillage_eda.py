import os

dataset_path = "PlantVillage"

classes = os.listdir(dataset_path)

print("Number of disease classes:", len(classes))

for c in classes:
    print(c, len(os.listdir(dataset_path + "/" + c)))