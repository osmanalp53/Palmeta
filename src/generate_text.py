import os
import sys
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
import re

# Yol ayarları
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_generator import CharTransformer, CHAR2IDX, IDX2CHAR, VOCAB_SIZE
from src.region_features import analyze_all_regions

# Model sabitleri
EMBED_DIM = 128
MAX_LEN = 100
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

region_names = ["venus", "jupiter", "saturn", "apollo", "mercury", "luna", "mars", "heart_center"]

# El görselinden vektör çıkar
def extract_features_from_image(image_path):
    img = imageio.imread(image_path, pilmode='L')
    img = img / 255.0
    features = analyze_all_regions(img)
    feature_vector = np.array([features[k] for k in region_names], dtype=np.float32)
    return feature_vector

# Promptlar (üretim dostu)
prompt_templates = {
    "venus": "Venüs çizgin duygusal yoğunluğunu yansıtıyor:",
    "jupiter": "Jüpiter çizgin özgüvenini nasıl etkiliyor:",
    "saturn": "Satürn çizgin sorumluluk alanını gösteriyor:",
    "apollo": "Apollo çizgin sanatsal yönünü yansıtıyor:",
    "mercury": "Merkür çizgin iletişim becerilerini nasıl etkiliyor:",
    "luna": "Luna alanın içsel dünyanı anlatıyor:",
    "mars": "Mars çizgin mücadele gücünü temsil ediyor:",
    "heart_center": "Kalp merkez çizgin duygusal dengen hakkında ne söylüyor:"
}

# Temizlik işlemleri
def clean_text(text):
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s\.,!?\'-]', '', text)
    return text

# Güzelleştirme + minimum 2 anlamlı cümle
def beautify_story(raw_text):
    raw_text = clean_text(raw_text.strip())
    if not raw_text.endswith('.'):
        raw_text += "."
    sentences = raw_text.split('.')
    sentences = [s.strip().capitalize() for s in sentences if len(s.split()) > 3 and len(s) > 10]
    return '. '.join(sentences[:2]) + '.' if sentences else "(Yorum üretilemedi.)"

# Top-k sampling destekli metin üretici
def generate_text(model, start_text, feature_vector=None, temperature=0.9, top_k=5, max_len=60):
    model.eval()
    generated = [CHAR2IDX.get(ch, 0) for ch in (start_text + " ").lower() if ch in CHAR2IDX]
    generated = generated[-MAX_LEN:]

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            logits = model(x, features=feature_vector)
            logits = logits[:, -1, :] / temperature
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_logits, dim=-1)
            next_idx = top_indices[0, torch.multinomial(probs, 1).item()].item()
            generated.append(next_idx)
            if IDX2CHAR[next_idx] == '\n':
                break
            if len(generated) > MAX_LEN:
                generated = generated[1:]

    return ''.join(IDX2CHAR[idx] for idx in generated)

# Ana program
if __name__ == "__main__":
    print(f"🚀 Başlatıldı | Cihaz: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")

    tk.Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Bir el görseli seçin", filetypes=[("Görüntü dosyaları", "*.png;*.jpg;*.jpeg")])

    if not image_path:
        print("❌ Hiçbir dosya seçilmedi.")
        exit()

    features = extract_features_from_image(image_path)
    torch_vector = torch.tensor(features).unsqueeze(0).to(DEVICE)
    print(f"🧬 El vektörü: {np.round(features, 3)}")

    model = CharTransformer(VOCAB_SIZE, EMBED_DIM, MAX_LEN, NUM_HEADS, NUM_LAYERS, FF_DIM).to(DEVICE)
    model.load_state_dict(torch.load("char_transformer.pth", map_location=DEVICE))
    print("✅ Model başarıyla yüklendi.")

    print("\n📜 8 Bölgeye Özel El Falı Yorumu:\n")
    for region in region_names:
        prompt = prompt_templates.get(region, f"{region.capitalize()} çizginin yorumu:") + " "
        raw = generate_text(model, prompt, feature_vector=torch_vector, temperature=0.9, top_k=5)
        final = beautify_story(raw)
        print(f"🔮 {prompt.strip()}\n{final}\n")

    print("🔚 Tüm bu izler, karakterini şekillendiren sessiz tanıklar gibi. Bu el, içsel dünyanı dışa vuran eşsiz bir harita gibi görünüyor.")
