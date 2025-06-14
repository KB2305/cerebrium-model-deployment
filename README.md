# 🧠 Cerebrium Model Deployment with FastAPI + ONNX

This project demonstrates deploying a deep learning model (ONNX format) via a FastAPI app, containerized using Docker, and optionally deployed on [Cerebrium](https://www.cerebrium.ai/).

---

## 📁 Project Structure

```
cerebrium-model-deployment/
├── app/
│ ├── app.py # FastAPI server
│ ├── model.py # Preprocessing and ONNX model loader
│ ├── convert_to_onnx.py # PyTorch → ONNX conversion script
│ ├── test.py # Local testing script
│ └── init.py
├── weights/
│ ├── model.onnx
│ └── pytorch_model_weights.pth
├── requirements.txt
├── Dockerfile
└── README.md
```


---

## 🚀 Run Locally

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server
```
uvicorn app.app:app --reload
```
Then open: http://127.0.0.1:8000/docs

## 🐳 Build & Run with Docker
### 1. Build Docker image
```
docker build -t my-model-app .
```
### 2.Run the container
```
docker run -p 8000:8000 my-model-app
```
## ☁️ Deploy on Cerebrium
### 1. Install Cerebrium CLI
```
pip install cerebrium --upgrade
```

### 2. Initialize project
```
cerebrium init my-project-name
```

### 3. Deploy
```
cerebrium deploy
```
Make sure to include your weights and cerebrium.toml file in the root directory.

## 📦 Requirements
Install all Python dependencies via:
```
pip install -r requirements.txt
```

## 📄 License
This project is for internal/company deployment via Cerebrium.

---

### ✅ **Step 2: Add, Commit & Push**

Once added:

```bash
git add README.md
git commit -m "Add README.md"
git push
