# Distributed Malware Family Classification using Binary Image Visualization

**Author:** Riley Meeves  
**Course:** SAT 5165 - Michigan Technological University  
**Date:** December 2025

---

## 📌 Project Overview
This project implements a distributed machine learning pipeline to classify malware families by visualizing their binary executables as grayscale images. 

Traditional malware detection relies on "signatures" (specific byte sequences), but new variants are produced daily, overwhelming these databases. This project leverages the theory that malware from the same family shares structural similarities in code organization. By mapping binary code to a 2D grid, these structures appear as distinct visual textures, allowing a Convolutional Neural Network (CNN) to classify them without reverse engineering.

### Key Features
* **Distributed Ingestion:** A PySpark pipeline processes raw `.bytes` files across a two-node cluster to prevent memory overhead.
* **Binary Visualization:** Converts hexadecimal code into $64 \times 64$ grayscale images.
* **Deep Learning:** A Sequential CNN trained on the generated images to classify 8 distinct malware families.

---

## ⚙️ Environment & Architecture

The project was designed for a two-node Apache Spark cluster running Linux (Fedora).

* **Master Node:** `hadoop1` (Running Spark Master) 
* **Worker Node:** `hadoop2` (Running Spark Worker) 
* **Spark Master URL:** `spark://hadoop1:7077`

### Dependencies
To replicate this environment, the following dependencies must be installed on all nodes:

* **System:** Python 3.x, Apache Spark 3.5.1
* **Python Libraries:**
    * `pyspark`
    * `numpy`
    * `pandas`
    * `tensorflow` (Keras)
    * `scikit-learn`
    * `pillow` (PIL)

---

## 🚀 Installation & Setup

1.  **Configure Networking:** Ensure `/etc/hosts` allows nodes to resolve each other by hostname (`hadoop1`, `hadoop2`).
2.  **SSH Access:** Configure passwordless SSH from the master to the worker node.
3.  **Data Placement:** Ensure the dataset and scripts are in the exact same path on both the master and worker nodes (e.g., `/home/sat3812/final_project`) as Spark executors read from the local filesystem.

---

## 💻 Usage

### 1. Data Preprocessing (Spark)
The `preprocess.py` script ingests the raw `.bytes` files, converts them to images, and saves them as Parquet files. It uses a Python `glob` approach to list files before parallelization to avoid JVM memory crashes.

```bash
# Submit the job to the Spark cluster
spark-submit --master spark://hadoop1:7077 preprocess.py
