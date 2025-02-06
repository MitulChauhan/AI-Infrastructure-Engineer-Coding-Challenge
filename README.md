# AI Product Matching System

## Overview

This project implements an end-to-end product matching system leveraging Vector Databases, NoSQL Databases, Visual Language Models (VLMs), and Vision Foundation Models served using NVIDIA Triton Inference Server.

## Features and Functionalities

1. **Vector Database (FAISS) for Nearest Neighbor Search**  
2. **MongoDB for Storing Product Metadata and Logs**  
3. **Model Quantization using TensorRT**  
4. **Deployment with NVIDIA Triton Inference Server**  
5. **Product Matching Pipeline (Image → Embeddings → Search → Metadata Retrieval)**  
6. **Supports Both Image and Text Queries**  
7. **Logging and Error Handling**  
8. **Containerized with Docker**  
9. **Batching and Caching for Optimized Latency**  
10. **Async Processing for Faster Inference**  
11. **Well-Structured, Modular Code with Documentation**  

---

## **Installation & Execution Guide**

### **1. Clone the Repository**

```sh
git clone https://github.com/MitulChauhan/AI-Infrastructure-Engineer-Coding-Challenge.git
cd ai-product-matching
```

### **2. Setup Virtual Environment & Install Dependencies**

```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Start MongoDB & FAISS Vector Database**

#### **Using Docker:**

```sh
docker run --name mongo-db -d -p 27017:27017 mongo
```

#### **Or manually (if MongoDB is installed):**

```sh
mongod --dbpath ./data/mongodb
```

### **4. Populate Database with Sample Product Data**

```sh
python scripts/populate_db.py
```

### **5. Quantize Model with TensorRT**

#### **Ensure TensorRT is Installed:**
- If using Docker:
  ```sh
  docker run --gpus all --rm -it nvcr.io/nvidia/tensorrt:latest bash
  ```
- If installing manually:
  ```sh
  pip install nvidia-pyindex
  pip install nvidia-tensorrt
  ```

#### **Run the Quantization Script:**
```sh
python scripts/quantize_model.py
```

#### **Verify the Quantized Model Exists:**
```sh
ls models/
```
- You should see `quantized_model.trt` inside the **models/** directory.

### **6. Start NVIDIA Triton Inference Server**

```sh
docker run --rm --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 --name triton-inference-server nvcr.io/nvidia/tritonserver:latest tritonserver --model-repository=/models
```

### **7. Run the Product Matching API**

```sh
uvicorn app.api:app --host 0.0.0.0 --port 5000 --reload
```

### **8. Test the API**

You can test the API using **Postman** or **cURL**.

#### **Upload Product Data**

```sh
curl -X POST "http://127.0.0.1:5000/upload" -H "Content-Type: application/json" -d '{"name": "Nike Shoes", "category": "Footwear", "price": 120, "image_path": "./images/nike.jpg"}'
```

#### **Match a Product Using an Image**

```sh
curl -X POST "http://127.0.0.1:5000/match" -H "Content-Type: application/json" -d '{"image_path": "./test_images/sample.jpg"}'
```

#### **Match a Product Using Text**

```sh
curl -X POST "http://127.0.0.1:5000/match" -H "Content-Type: application/json" -d '{"query_text": "red running shoes"}'
```

#### **Check Logs**

```sh
curl -X GET "http://127.0.0.1:5000/logs"
```

### **9. Run the System in Docker (Optional)**

#### **Build and Run the Docker Container**

```sh
docker build -t ai-product-matching .
docker run -p 5000:5000 ai-product-matching
```

---

For any questions, feel free to reach me out through email: chauhanmitul1301@gmail.com!
