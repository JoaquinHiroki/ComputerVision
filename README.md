
# 📷 License Plate Recognition and Tracker

Este proyecto implementa un sistema de detección y seguimiento de placas vehiculares en tiempo real usando **OpenCV**, **EasyOCR** y filtros personalizados de procesamiento de imágenes.

## 🚀 Características

- Detección de texto en placas con EasyOCR.
- Filtros de imagen personalizados para mejorar la detección (blanco y negro, contraste).
- Seguimiento de entrada y salida de vehículos con registro en archivo CSV.
- Detección en vivo desde cámara web o archivo de video.
- Visualización de placas detectadas y su estado (dentro/fuera).

## 🧠 Tecnologías usadas

- Python 3
- OpenCV
- EasyOCR
- NumPy
- CSV / manejo de archivos
- Terminal (modo script interactivo)


## 🧠 How It Works: EasyOCR + OpenCV

This project uses [**EasyOCR**](https://github.com/JaidedAI/EasyOCR) and [**OpenCV**](https://opencv.org/) to detect and extract text from images. Here's a quick overview of what each library does and how they work together:

### 📷 OpenCV: Image Processing
OpenCV (Open Source Computer Vision Library) is used to:
- **Load and preprocess images** (e.g., resizing, converting to grayscale, thresholding).
- **Enhance text regions** for better OCR results (e.g., noise reduction, edge detection, morphological transformations).
- Optionally **detect contours or regions of interest** before OCR to reduce false positives.

### 🔍 EasyOCR: Text Recognition
EasyOCR is a deep learning–based Optical Character Recognition (OCR) library that:
- Uses a **pretrained neural network** to read text in multiple languages directly from images.
- Supports detection of printed and handwritten text.
- Works out of the box with minimal configuration.

# ComputerVision
