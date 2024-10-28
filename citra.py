import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os  # Menambahkan impor os untuk mengelola file

# Fungsi untuk menampilkan gambar dengan konversi BGR ke RGB
def tampilkan_judul(citra, judul):
    if len(citra.shape) == 3:  # Citra berwarna (BGR)
        citra = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)
    st.image(citra, caption=judul, use_column_width=True)

# Fungsi untuk membuat dan menampilkan histogram sebagai bar plot
def tampilkan_histogram(citra):
    fig, ax = plt.subplots()
    if len(citra.shape) == 3:  # Histogram untuk gambar berwarna (BGR)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([citra], [i], None, [256], [0, 256])
            ax.bar(np.arange(256), hist.ravel(), color=col, alpha=0.5, width=1.0)
        ax.set_title('Histogram (BGR)')
    else:  # Histogram untuk gambar grayscale
        hist = cv2.calcHist([citra], [0], None, [256], [0, 256])
        ax.bar(np.arange(256), hist.ravel(), color='black', alpha=0.7, width=1.0)
        ax.set_title('Histogram (Grayscale)')
    ax.set_xlim([0, 256])
    st.pyplot(fig)

# Fungsi untuk menyimpan citra hasil ke file lokal
def simpan_citra(citra, nama_file):
    cv2.imwrite(nama_file, citra)
    st.success(f"Citra berhasil disimpan sebagai {nama_file}")

# Judul Aplikasi
st.title("Aplikasi Pengolahan Citra")

# Input Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar dengan OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

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
    def olah_gambar(img, opsi):
        if opsi == "Citra Asli":
            return img
        elif opsi == "Citra Negatif":
            return 255 - img
        elif opsi == "Grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif opsi == "Rotasi 90 Derajat":
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
            return cv2.warpAffine(img, M, (w, h))
        elif opsi == "Histogram Equalization":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.equalizeHist(gray)
        elif opsi == "Black & White":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return bw
        elif opsi == "Smoothing (Gaussian Blur)":
            return cv2.GaussianBlur(img, (5, 5), 0)

    # Hasil pemrosesan untuk dua mode
    hasil1 = olah_gambar(img, opsi1)
    hasil2 = olah_gambar(img, opsi2)

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
