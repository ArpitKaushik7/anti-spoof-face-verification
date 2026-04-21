import os
from PIL import Image

base_dir = "../data"
deleted = 0
checked = 0

for folder in ["real", "spoof"]:
    path = os.path.join(base_dir, folder)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        try:
            img = Image.open(file_path)
            img.verify()
            checked += 1
        except Exception:
            print("Removing corrupted file:", file_path)
            os.remove(file_path)
            deleted += 1

print("Checked:", checked)
print("Deleted:", deleted)