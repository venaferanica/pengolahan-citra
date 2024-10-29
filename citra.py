import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
import os
from io import BytesIO

# Fungsi untuk menampilkan gambar dengan judul
def tampilkan_judul(citra, judul):
    st.image(citra, caption=judul, use_column_width=True)

# Fungsi untuk menampilkan histogram sebagai bar plot
def tampilkan_histogram(citra):
    fig, ax = plt.subplots()
    if len(citra.shape) == 3:  # Gambar berwarna
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = np.histogram(citra[:, :, i], bins=256, range=(0, 256))[0]
            ax.bar(np.arange(256), hist, color=col, alpha=0.5, width=1.0)
        ax.set_title('Histogram (RGB)')
    else:  # Gambar grayscale
        hist, _ = np.histogram(citra.flatten(), bins=256, range=(0, 256))
        ax.bar(np.arange(256), hist, color='black', alpha=0.7, width=1.0)
        ax.set_title('Histogram (Grayscale)')
    ax.set_xlim([0, 256])
    st.pyplot(fig)

# Konversi array numpy ke bytes untuk diunduh
def convert_image_to_bytes(image_array):
    img = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# Fungsi untuk mengolah gambar berdasarkan opsi
def olah_gambar(img_np, opsi, threshold=127, rotasi=90):
    if opsi == "Citra Negatif":
        return 255 - img_np
    elif opsi == "Grayscale":
        return np.array(ImageOps.grayscale(Image.fromarray(img_np)))
    elif opsi == "Rotasi":
        return np.rot90(img_np, k=rotasi // 90)
    elif opsi == "Histogram Equalization":
        r, g, b = Image.fromarray(img_np).split()
        return np.array(Image.merge("RGB", [ImageOps.equalize(c) for c in (r, g, b)]))
    elif opsi == "Black & White":
        gray = np.array(ImageOps.grayscale(Image.fromarray(img_np)))
        return np.where(gray > threshold, 255, 0).astype(np.uint8)
    elif opsi == "Smoothing (Gaussian Blur)":
        return np.array(Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=2)))

# Judul Aplikasi
st.title("Pengolahan Citra Kelompok Esigma")

# Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    st.subheader("Gambar Asli dan Histogram")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(img_np, "Gambar Asli")
    with col2:
        tampilkan_histogram(img_np)

    st.sidebar.subheader("Pilih Mode Pengolahan Citra")
    opsi = st.sidebar.selectbox("Mode Pengolahan", [
        "Citra Negatif", "Grayscale", "Rotasi", 
        "Histogram Equalization", "Black & White", 
        "Smoothing (Gaussian Blur)"
    ])

    if opsi == "Black & White":
        threshold = st.sidebar.slider("Threshold", 0, 255, 127)
    elif opsi == "Rotasi":
        rotasi = st.sidebar.radio("Derajat Rotasi", [90, 180, 270])

    hasil = olah_gambar(img_np, opsi, threshold, rotasi)

    st.subheader(f"Hasil - {opsi}")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(hasil, f"Hasil - {opsi}")
    with col2:
        tampilkan_histogram(hasil)

    nama_file = f"{os.path.splitext(uploaded_file.name)[0]}-{opsi.lower().replace(' ', '_')}.png"
    st.download_button(
        label=f"Download {opsi}",
        data=convert_image_to_bytes(hasil),
        file_name=nama_file,
        mime="image/png"
    )
else:
    st.write("Silakan upload gambar terlebih dahulu.")
