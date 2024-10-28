import os

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageOps


# Fungsi untuk menampilkan gambar
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

# Fungsi untuk menyimpan citra hasil ke file lokal
def simpan_citra(citra, nama_file):
    Image.fromarray(citra.astype(np.uint8)).save(nama_file)
    st.success(f"Citra berhasil disimpan sebagai {nama_file}")

# Judul Aplikasi
st.title("Pengolahan Citra Kelompok Esigma")

# Input Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar dengan Pillow
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    # Dapatkan nama file asli
    original_filename = uploaded_file.name

    # Sidebar untuk memilih dua mode untuk dibandingkan
    st.sidebar.subheader("Pilih Mode Pengolahan Citra")
    opsi1 = st.sidebar.selectbox("Mode 1", (
        "Citra Asli",
        "Citra Negatif",
        "Grayscale",
        "Rotasi 90 Derajat",
        "Histogram Equalization",
        "Black & White",
        "Smoothing (Gaussian Blur)"
    ))
    opsi2 = st.sidebar.selectbox("Mode 2", (
        "Citra Asli",
        "Citra Negatif",
        "Grayscale",
        "Rotasi 90 Derajat",
        "Histogram Equalization",
        "Black & White",
        "Smoothing (Gaussian Blur)"
    ))

    # Fungsi untuk mengolah gambar berdasarkan opsi
    def olah_gambar(img_np, opsi):
        if opsi == "Citra Asli":
            return img_np
        elif opsi == "Citra Negatif":
            return 255 - img_np
        elif opsi == "Grayscale":
            return np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
        elif opsi == "Rotasi 90 Derajat":
            return np.rot90(img_np, 1)
        elif opsi == "Histogram Equalization":
            gray = np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
            return np.array(ImageOps.equalize(Image.fromarray(gray.astype(np.uint8))))
        elif opsi == "Black & White":
            gray = np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
            bw = np.where(gray > 127, 255, 0).astype(np.uint8)
            return bw
        elif opsi == "Smoothing (Gaussian Blur)":
            return np.array(Image.fromarray(img_np.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)))

    # Hasil pemrosesan untuk dua mode
    hasil1 = olah_gambar(img_np, opsi1)
    hasil2 = olah_gambar(img_np, opsi2)

    # Menampilkan gambar dan histogram untuk kedua mode
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(hasil1, f"Hasil - {opsi1}")
        tampilkan_histogram(hasil1)

    with col2:
        tampilkan_judul(hasil2, f"Hasil - {opsi2}")
        tampilkan_histogram(hasil2)

        # Tombol untuk menyimpan hasil 2
        if st.button(f"Simpan {opsi2}", key=f"simpan_{opsi2}_2"):
            # Ubah ekstensi berdasarkan format file asli
            ext = os.path.splitext(original_filename)[1]  # Mendapatkan ekstensi file asli
            nama_file_simpan = f"{os.path.splitext(original_filename)[0]}-{opsi2.lower().replace(' ', '_')}{ext}"
            simpan_citra(hasil2, nama_file_simpan)

else:
    st.write("Silakan upload gambar terlebih dahulu.")
