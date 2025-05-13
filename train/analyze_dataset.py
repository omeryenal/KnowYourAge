import os
import matplotlib.pyplot as plt

BASE_DIR = "../data/UTKFace"
ages = []

for part_name in os.listdir(BASE_DIR):
    part_path = os.path.join(BASE_DIR, part_name)
    if not os.path.isdir(part_path):
        continue

    for filename in os.listdir(part_path):
        if filename.endswith(".jpg"):
            try:
                age = int(filename.split("_")[0])
                ages.append(age)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

plt.figure(figsize=(10, 6))
plt.hist(ages, bins=range(0, 101, 5), edgecolor='black')
plt.title("Age Distribution in UTKFace (All Parts)")
plt.xlabel("Age")
plt.ylabel("Number of Images")
plt.grid(True)
plt.show()
