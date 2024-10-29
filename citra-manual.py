import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
import os
from io import BytesIO

# Fungsi untuk menampilkan gambar dengan judul
def tampilkan_judul(citra, judul):
    st.image(citra, caption=judul, use_column_width=True)

# Fungsi untuk membuat dan menampilkan histogram sebagai bar plot
def tampilkan_histogram(citra):
    fig, ax = plt.subplots()
    if len(citra.shape) == 3:  # Histogram untuk gambar berwarna
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = np.histogram(citra[:, :, i], bins=256, range=(0, 256))[0]
            ax.bar(np.arange(256), hist, color=col, alpha=0.5, width=1.0)
        ax.set_title('Histogram (RGB)')
    else:  # Histogram untuk gambar grayscale
        hist, _ = np.histogram(citra.flatten(), bins=256, range=(0, 256))
        ax.bar(np.arange(256), hist, color='black', alpha=0.7, width=1.0)
        ax.set_title('Histogram (Grayscale)')
    ax.set_xlim([0, 256])
    st.pyplot(fig)

# Fungsi untuk mengkonversi array numpy menjadi bytes
def convert_image_to_bytes(image_array):
    img = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

# Fungsi untuk histogram equalization
def histogram_equalization(img_np):
    img_flat = img_np.flatten()
    hist, bins = np.histogram(img_flat, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    img_eq = np.interp(img_flat, bins[:-1], cdf_normalized).reshape(img_np.shape)
    return img_eq.astype(np.uint8)

# Fungsi untuk konversi ke hitam-putih
def to_black_and_white(img_np, threshold):
    gray = np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
    bw = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return bw

# Fungsi untuk Gaussian Blur
def gaussian_blur(img_np, radius):
    # Create Gaussian kernel
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    sigma = radius / 3.0
    total = 0.0

    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            value = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[x + radius, y + radius] = value
            total += value

    # Normalize the kernel
    kernel /= total

    # Pad the image to handle borders
    pad_width = radius
    padded_img = np.pad(img_np, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='edge')
    blurred_img = np.zeros_like(img_np)

    # Apply Gaussian kernel to each pixel
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            for c in range(img_np.shape[2]):
                blurred_img[i, j, c] = np.sum(
                    kernel * padded_img[i:i + size, j:j + size, c]
                )

    return blurred_img.astype(np.uint8)

# Fungsi untuk konversi ke grayscale
def to_grayscale(img_np):
    return np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))

# Fungsi untuk memilih channel RGB
def select_channel(img_np, channel):
    img_channel = np.zeros_like(img_np)
    channel_map = {"Red": 0, "Green": 1, "Blue": 2}
    img_channel[:, :, channel_map[channel]] = img_np[:, :, channel_map[channel]]
    return img_channel

# Judul Aplikasi
st.title("Pengolahan Citra Kelompok Esigma")

# Input Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar dengan Pillow
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    # Menampilkan gambar dan histogram asli
    st.subheader("Gambar Asli dan Histogram")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(img_np, "Gambar Asli")
    with col2:
        tampilkan_histogram(img_np)

    # Sidebar untuk memilih mode pemrosesan gambar
    st.sidebar.subheader("Pilih Mode Pengolahan Citra")
    opsi = st.sidebar.selectbox("Mode Pengolahan", (
        "Citra Negatif", "Grayscale", "Rotasi", 
        "Histogram Equalization", "Black & White", "Smoothing (Gaussian Blur)", "Channel RGB"
    ))

    # Input untuk threshold jika opsi "Black & White" dipilih
    if opsi == "Black & White":
        threshold = st.sidebar.number_input("Threshold Level", min_value=0, max_value=255, value=127)

    # Button untuk memilih derajat rotasi jika opsi "Rotasi" dipilih
    if opsi == "Rotasi":
        rotasi = st.sidebar.radio("Pilih Derajat Rotasi", (90, 180, 270))

    # Field input untuk blur radius jika opsi "Smoothing (Gaussian Blur)" dipilih
    if opsi == "Smoothing (Gaussian Blur)":
        blur_radius = st.sidebar.text_input("Masukkan Blur Radius", value="10")
        try:
            blur_radius = float(blur_radius)
        except ValueError:
            st.sidebar.error("Masukkan nilai numerik yang valid untuk blur radius.")
            blur_radius = 10  # Default value jika input salah

    # Pilihan channel jika opsi "Channel RGB" dipilih
    if opsi == "Channel RGB":
        channel = st.sidebar.selectbox("Pilih Channel", ("Red", "Green", "Blue"))

    # Fungsi untuk mengolah gambar berdasarkan opsi
    def olah_gambar(img_np, opsi):
        if opsi == "Citra Negatif":
            return np.clip(255 - img_np.astype(np.uint8), 0, 255)
        elif opsi == "Grayscale":
            return to_grayscale(img_np)
        elif opsi == "Rotasi":
            if rotasi == 90:
                return np.rot90(img_np, 1)
            elif rotasi == 180:
                return np.rot90(img_np, 2)
            elif rotasi == 270:
                return np.rot90(img_np, 3)
        elif opsi == "Histogram Equalization":
            return histogram_equalization(img_np)
        elif opsi == "Black & White":
            return to_black_and_white(img_np, threshold)
        elif opsi == "Smoothing (Gaussian Blur)":
            return gaussian_blur(img_np, blur_radius)
        elif opsi == "Channel RGB":
            return select_channel(img_np, channel)

    # Pemrosesan gambar berdasarkan opsi
    hasil = olah_gambar(img_np, opsi)

    # Menampilkan hasil pemrosesan dan histogram
    st.subheader(f"Hasil - {opsi}")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(hasil, f"Hasil - {opsi}")
    with col2:
        tampilkan_histogram(hasil)

    # Membuat nama file untuk hasil yang akan diunduh
    original_filename = uploaded_file.name
    ext = os.path.splitext(original_filename)[1]
    nama_file_simpan = f"{os.path.splitext(original_filename)[0]}-{opsi.lower().replace(' ', '_')}{ext}"

    # Konversi hasil menjadi bytes
    hasil_bytes = convert_image_to_bytes(hasil)

    # Tombol download
    st.download_button(
        label=f"Download {opsi}",
        data=hasil_bytes,
        file_name=nama_file_simpan,
        mime=f"image/{ext[1:]}"
    )

else:
    st.write("Silakan upload gambar terlebih dahulu.")
