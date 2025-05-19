import numpy as np
from skimage.transform import resize  # Görsel yeniden boyutlandırma için

def resize_image(img, size=(256, 256)):
    """
    Görseli yeniden boyutlandırır (256x256).
    """
    return resize(img, size, preserve_range=True, anti_aliasing=True).astype(np.float32)

def convolve(img, kernel):
    """
    Basit evrişim (convolution) işlemi.
    """
    h, w = img.shape
    kh, kw = kernel.shape
    padded = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    result = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result

def sobel_edges(img):
    """
    Kenar çıkarımı için sobel filtresi.
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude

def get_region(img, coords_ratio):
    """
    Görselden oransal bölge kırp.
    coords_ratio: (x1, y1, x2, y2) oranları (0-1 arasında)
    """
    h, w = img.shape
    x1 = int(coords_ratio[0] * w)
    y1 = int(coords_ratio[1] * h)
    x2 = int(coords_ratio[2] * w)
    y2 = int(coords_ratio[3] * h)
    return img[y1:y2, x1:x2]

def analyze_region(region_img):
    """
    Bölgedeki kenar yoğunluğunu hesapla.
    """
    edges = sobel_edges(region_img)
    binary = edges > 0.3
    density = np.sum(binary) / binary.size
    return density

def analyze_all_regions(img):
    """
    Tüm el bölgelerinden yoğunluk çıkar ve sözlük olarak döndür.
    """
    resized = resize_image(img)

    region_defs = {
        "venus": (0.05, 0.65, 0.35, 0.95),
        "jupiter": (0.15, 0.05, 0.30, 0.20),
        "saturn": (0.40, 0.05, 0.60, 0.20),
        "apollo": (0.65, 0.05, 0.80, 0.20),
        "mercury": (0.82, 0.05, 0.95, 0.20),
        "luna": (0.70, 0.65, 0.95, 0.95),
        "mars": (0.40, 0.40, 0.60, 0.60),
        "heart_center": (0.30, 0.25, 0.70, 0.35)
    }

    results = {}
    for name, coords in region_defs.items():
        region = get_region(resized, coords)
        density = analyze_region(region)
        results[name] = round(float(density), 3)

    return results
