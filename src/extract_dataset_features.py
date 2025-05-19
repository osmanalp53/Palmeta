import os
import imageio.v2 as imageio
import numpy as np
import csv
from region_features import analyze_all_regions

# 📁 Görsel klasörü ve CSV dosyası yolu
image_dir = os.path.join(os.path.dirname(__file__), "..", "data", "hand_images")
output_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed_vectors", "region_features.csv")

# 📌 Bölgelerin isimleri
region_names = ["venus", "jupiter", "saturn", "apollo", "mercury", "luna", "mars", "heart_center"]

# 📤 CSV dosyasını oluştur ve yaz
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name"] + region_names)

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, file_name)
            try:
                img = imageio.imread(image_path, pilmode='L')
                img = img / 255.0
                features = analyze_all_regions(img)
                row = [file_name] + [features[region] for region in region_names]
                writer.writerow(row)
                print(f"✅ {file_name} işlendi")
            except Exception as e:
                print(f"❌ Hata ({file_name}): {e}")
