# Fruit Freshness & Type Classifier 🍎🍌🍓

A **deep learning application** for classifying fruits by **type** and **freshness**, built with **Convolutional Neural Networks (CNNs)** and an interactive **Streamlit** interface. Train new models, make predictions, and manage saved models easily through the web app.  

---

## 🚀 Features

- **Fruit Classification**  
  Identify fruit types: **apple, banana, strawberry**.

- **Freshness Detection**  
  Detect whether a fruit is **fresh** or **rotten**.

- **Interactive Streamlit Interface**  
  - Upload single images for inference.  
  - Upload datasets in ZIP format to train new models.  
  - Load, select, and delete saved models.  
  - Monitor training progress in real-time with progress bars.  

- **Multiple Model Support**  
  Save multiple trained models and name them for future use.

---

## 🗂 Project Structure
```
project-root/
│
├── models/ # Saved model weights (*.weights.h5)
├── requirements.txt # Python dependencies
├── untitled20.py # Streamlit application
└── README.md # Project documentation
```
---

## 📁 Dataset Format

```
dataset/
├── apple/
│ ├── fresh/
│ └── rotten/
├── banana/
│ ├── fresh/
│ └── rotten/
└── strawberry/
├── fresh/
└── rotten/
```
Downloaded from Kaggle using:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdulrafeyyashir/fresh-vs-rotten-fruit-images")

print("Path to dataset files:", path)
```


<img width="839" height="601" alt="image" src="https://github.com/user-attachments/assets/d6faf995-62fc-459e-a3e6-0cce8c41355b" />
<img width="840" height="615" alt="image" src="https://github.com/user-attachments/assets/cc935bed-dda4-48ae-8a20-43a9e13336f2" />


