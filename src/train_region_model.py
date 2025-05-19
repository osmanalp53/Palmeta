import csv
import numpy as np

# 1. Veriyi yükle
def load_data(csv_path):
    X = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # başlıkları atla
        for row in reader:
            values = list(map(float, row[1:]))  # sadece 8 bölgesel değer
            X.append(values)
    return np.array(X)

# 2. Basit bir sinir ağı sınıfı (tamamen numpy ile)
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim,))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.z1 = self.relu(np.dot(self.W1, x.T).T + self.b1)
        self.out = np.dot(self.W2, self.z1.T).T + self.b2
        return self.out

    def train(self, X, epochs=500, lr=0.01):
        for epoch in range(epochs):
            # Forward
            preds = self.forward(X)

            # Loss (Mean Squared Error)
            loss = np.mean((preds - X) ** 2)

            # Backprop
            d_out = 2 * (preds - X) / X.shape[0]
            dW2 = np.dot(d_out.T, self.z1)
            db2 = np.sum(d_out, axis=0)

            d_hidden = np.dot(d_out, self.W2)
            d_hidden[self.z1 <= 0] = 0  # ReLU grad
            dW1 = np.dot(d_hidden.T, X)
            db1 = np.sum(d_hidden, axis=0)

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 50 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

# 3. Eğitim başlat
if __name__ == "__main__":
    csv_path = "C:/Users/osman/Desktop/Palmeta/data/processed_vectors/region_features.csv"

    X = load_data(csv_path)

    print(f"📊 Eğitim verisi boyutu: {X.shape}")  # (örneğin: 1000, 8)

    model = MLP(input_dim=8, hidden_dim=16, output_dim=8)
    model.train(X, epochs=500, lr=0.01)

    print("✅ Model eğitim tamamlandı.")
