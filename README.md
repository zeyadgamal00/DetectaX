# 🚗 CompCars Vision App  
### **Fine-Grained Car Classification & Object Recognition Web Application**  
**“Upload. Detect. Classify. Explore.”**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📸 Demo  
> Replace the placeholders below with your actual screenshots or GIFs.

![Demo GIF](path/to/demo.gif)  
![Classification Screenshot](path/to/classification.png)  
![Detection Screenshot](path/to/detection.png)

---

## 📘 About the Project

The **CompCars Vision App** is a Streamlit-based web application built for performing **two powerful computer vision tasks**:

1. **Fine-Grained Car Classification**  
   Upload an image, and the model predicts **car make, model, and year** using a pretrained **ResNet50** architecture fine-tuned on the **CUHK-CompCars dataset**.

2. **Object Recognition & Detection**  
   The app uses **YOLO 12X**, a state-of-the-art object detection model, to identify and localize objects within an image, drawing bounding boxes with confidence scores.

This project is designed for **researchers, automotive analytics teams, ML students**, and anyone working with vehicle datasets who needs a clean, fast, interactive interface.

---

## 🧰 Tech Stack

| Category | Technologies |
|---------|--------------|
| **Frontend** | Streamlit |
| **Deep Learning** | ResNet50 (Image Classification), YOLO 12X (Object Detection) |
| **Computer Vision** | OpenCV |
| **Core Python Libraries** | NumPy, Pandas, Pillow |
| **Model Serving / Utils** | TensorFlow or PyTorch (depending on your implementation) |
| **Deployment** | Local machine or cloud (Streamlit Cloud, Azure, GCP, etc.) |

---

## ⭐ Features

### **🟦 Image Classification**
- Predicts **car make**
- Predicts **car model**
- Predicts **manufacturing year**
- Fine-grained classification powered by ResNet50
- Trained using CUHK-CompCars dataset  
- Supports JPG, PNG uploads

### **🟥 Object Recognition (YOLO 12X)**
- Detects multiple objects in an image  
- Draws bounding boxes with labels & confidence  
- Fast inference  
- High-accuracy detection for real-world scenes  

---

## Model Details

### **ResNet50 — Car Classification**
- Pretrained on ImageNet  
- Fine-tuned on CUHK-CompCars  
- Excellent at capturing **fine-grained automotive features**  
- Predicts: **Make → Model → Year**

### **YOLO 12X — Object Recognition**
- Cutting-edge YOLO variant  
- Optimized for high-speed detection  
- Generates precise bounding boxes and class names  

---

## Installation & Usage

Follow these steps to run the project locally:

---

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/compcars-vision-app.git
cd compcars-vision-app
```

### **2. Create a Virtual Environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```


Mac/Linux:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py