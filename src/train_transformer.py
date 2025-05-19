import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# 🔧 Yol ayarı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_generator import CharTransformer, CHAR2IDX, VOCAB_SIZE

# ⚙️ Hiperparametreler
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📄 Dosya yolları
corpus_path = r"C:\Users\osman\Desktop\Palmeta\data\corpus.txt"
feature_path = r"C:\Users\osman\Desktop\Palmeta\data\processed_vectors\region_features.csv"
model_path = r"C:\Users\osman\Desktop\Palmeta\char_transformer.pth"

# 🧹 Dataset
class TextWithFeaturesDataset(Dataset):
    def __init__(self, corpus_path, feature_path, seq_len=MAX_LEN):
        self.seq_len = seq_len
        self.samples = []

        df = pd.read_csv(feature_path)
        feature_map = {row["image_name"]: row.drop("image_name").values.astype("float32") for _, row in df.iterrows()}

        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for i in range(0, len(lines), 2):
                tag_line = lines[i].strip()
                text_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if not tag_line.startswith("[") or not text_line:
                    continue
                filename = tag_line.replace("[", "").replace("]", "").split(",")[0].strip() + ".jpg"
                if filename in feature_map:
                    text = ''.join(ch for ch in text_line.lower() if ch in CHAR2IDX)
                    encoded = [CHAR2IDX[ch] for ch in text]
                    for j in range(0, len(encoded) - seq_len):
                        x = encoded[j:j + seq_len]
                        y = encoded[j + 1:j + seq_len + 1]
                        self.samples.append((torch.tensor(x), torch.tensor(y), torch.tensor(feature_map[filename])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 🧠 Eğitim fonksiyonu
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y, features) in enumerate(dataloader):
        x, y, features = x.to(device), y.to(device), features.to(device)
        optimizer.zero_grad()
        logits = model(x, features=features)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        percent = 100 * (batch_idx + 1) / len(dataloader)
        print(f"📦 Batch {batch_idx+1}/{len(dataloader)} | İlerleme: {percent:.1f}% | Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# 🚀 Ana akış
if __name__ == "__main__":
    print(f"🚀 Eğitim başlıyor | Cihaz: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")

    dataset = TextWithFeaturesDataset(corpus_path, feature_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CharTransformer(VOCAB_SIZE, EMBED_DIM, MAX_LEN, NUM_HEADS, NUM_LAYERS, FF_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch {epoch + 1}/{EPOCHS}")
        avg_loss = train(model, dataloader, optimizer, criterion, DEVICE)
        print(f"✅ Epoch {epoch + 1} tamamlandı | Ortalama Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print("📎 Model başarıyla kaydedildi.")
                    