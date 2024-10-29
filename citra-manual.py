import streamlit as st
import os
from io import BytesIO

# Fungsi untuk membaca file gambar sebagai byte
def read_image_bytes(uploaded_file):
    return uploaded_file.read()

# Fungsi untuk menghitung citra negatif secara manual
def citra_negatif(image_bytes):
    # Citra negatif akan dihasilkan dengan membalikkan nilai pixel
    return bytes([255 - byte for byte in image_bytes])

# Fungsi untuk mengkonversi citra ke grayscale
def grayscale(image_bytes):
    # Asumsikan format gambar adalah RGB, setiap pixel terdiri dari 3 byte
    grayscale_bytes = bytearray()
    for i in range(0, len(image_bytes), 3):
        r = image_bytes[i]
        g = image_bytes[i + 1]
        b = image_bytes[i + 2]
        # Hitung nilai grayscale menggunakan rumus luminance
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
        grayscale_bytes.extend([gray] * 3)  # Ubah ke RGB dengan nilai yang sama
    return bytes(grayscale_bytes)

# Judul Aplikasi
st.title("Pengolahan Citra Tanpa Library")

# Input Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar sebagai bytes
    image_bytes = read_image_bytes(uploaded_file)

    # Menampilkan gambar asli
    st.image(image_bytes, caption="Gambar Asli", use_column_width=True)

    # Sidebar untuk memilih mode pemrosesan gambar
    st.sidebar.subheader("Pilih Mode Pengolahan Citra")
    opsi = st.sidebar.selectbox("Mode Pengolahan", ("Citra Negatif", "Grayscale"))

    # Pemrosesan gambar berdasarkan opsi
    if opsi == "Citra Negatif":
        hasil_bytes = citra_negatif(image_bytes)
        hasil_caption = "Hasil - Citra Negatif"
    elif opsi == "Grayscale":
        hasil_bytes = grayscale(image_bytes)
        hasil_caption = "Hasil - Grayscale"

    # Menampilkan hasil pemrosesan
    st.subheader(hasil_caption)
    st.image(hasil_bytes, caption=hasil_caption, use_column_width=True)

    # Membuat nama file untuk hasil yang akan diunduh
    original_filename = uploaded_file.name
    ext = os.path.splitext(original_filename)[1]
    nama_file_simpan = f"{os.path.splitext(original_filename)[0]}-{opsi.lower().replace(' ', '_')}{ext}"

    # Tombol download
    st.download_button(
        label=f"Download {opsi}",
        data=hasil_bytes,
        file_name=nama_file_simpan,
        mime=f"image/{ext[1:]}"
    )

else:
    st.write("Silakan upload gambar terlebih dahulu.")