# 🌱 AgriVision AI

An end-to-end deep learning application for **plant disease detection** using **PyTorch** and **ResNet18 Transfer Learning**.

## 🚀 Features

- Detects **15 plant disease categories**
- Trained on the **PlantVillage** dataset (20,638 images)
- Built using **PyTorch** and **Torchvision**
- Uses **Transfer Learning** with **ResNet18**
- Interactive prediction interface using **Gradio**
- Supports Apple Silicon (MPS) acceleration

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| Dataset | PlantVillage |
| Images | 20,638 |
| Classes | 15 |
| Architecture | ResNet18 |
| Validation Accuracy | **93.60%** |

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Gradio
- Pillow

---

## 📂 Project Structure

```
AgriVision-AI
│
├── app.py
├── predict.py
├── train.py
├── models/
├── utils/
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/Veritaserum04/AgriVision-AI.git

cd AgriVision-AI

pip install -r requirements.txt

python app.py
```

---

## 📚 Dataset

PlantVillage Dataset

- 20,638 labeled leaf images
- 15 disease categories

---

## 🎯 Future Improvements

- Fine-tune deeper ResNet layers
- Deploy online
- Add disease descriptions and treatment recommendations
- Improve prediction interface

---

## 👩‍💻 Author

**Amrutha V**
