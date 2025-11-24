# 🎵 Hybrid Music Recommender System
## User–Genre ALS + Content-Based Filtering (CBF) Hybrid Model

This project builds a hybrid music recommendation system using a sampled portion of the **Million Song Dataset (MSD)**.  
It combines:

- 🎧 **Collaborative Filtering (ALS)** — predicts user preferred genres
- 🔍 **Content-Based Filtering (CBF)** — models user taste based on actual audio features

By blending behavior and content, the system produces more balanced and genre-consistent recommendations.

---

## 📌 Table of Contents
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)

---

## 🚀 Key Features

### 🎼 MSD Data Preprocessing & Sampling
- Filters users and tracks from **train_triplets.txt**
- Extracts song-level audio features (tempo, loudness, year, etc.)
- Loads artist tags from MSD HDF5 metadata
- Maps tags into **15+ consolidated genres**
- Generates final content table `df_content_final`

---

### 🤖 User–Genre ALS Model
- Converts User–Item → **User–Genre** matrix  
- Trains a genre-level ALS model using the `implicit` library  
- Computes latent-factor genre preference scores  
- Saves **Top-2 preferred genres** for each user  
- Applies rare-genre weighting for balance  

---

### 🎧 Content-Based User Profiles (CBF)
- Builds user vectors by averaging features of all songs they listened to  
- Computes cosine similarity for scoring candidate tracks  

---

### 🔀 Hybrid Scoring Strategy
- Base score = **CBF similarity**
- Applies **0.4 penalty** to tracks whose genres mismatch the user's top genres  
- Removes previously listened tracks  
- Produces final Top-K recommendations  

---

## 🧱 Architecture

### **1. Data Preparation — `msd_data_sample.py`**

**Inputs**
- `train_triplets.txt`
- MSD Subset HDF5 files

**Process**
- User & track filtering  
- Feature extraction (audio + tags)  
- Genre grouping into 15+ categories  
- Train/test split → `R_train`, `R_test`  
- Generate `df_content_final`

---

### **2. User–Genre ALS Training — `model_genre_cf.py`**

Steps:
1. Build the User–Genre interaction matrix  
2. Re-weight rare genres  
3. Train ALS  
4. Compute latent genre preference scores  
5. Save Top-2 genres → `user_top_genres.pkl`

---

### **3. Hybrid Recommendation & Evaluation — `model_hybrid_final_final.py`**

- Create CBF user profiles  
- Compute similarity scores  
- Apply genre mismatch penalty  
- Generate Top-K recommendations  
- Evaluate using Precision@K, Recall@K, and cosine-based genre similarity  

---

## 📂 Data Requirements
Required files:

data install link : https://drive.google.com/file/d/1pdeyObIWA_qY4UfXxLxATVgnR3hghyO1/view?usp=sharing

1.  **Listening History:**
    *   `data/train_triplets.txt`
2.  **Song Metadata/Analysis Data:**
    *   The **Million Song Subset** directory structure containing HDF5 files (e.g., `data/MillionSongSubset/`).
    *   
## 🛠️ Installation

Install the necessary Python libraries using pip:

```bash
pip install pandas numpy scipy implicit scikit-learn h5py
```

## ▶️ How to Run

The project must be executed in the following sequential order. 
Each step generates necessary intermediate files in the data/ directory for the next step.

### Step 1: Data Preprocessing and Feature Engineering
```bash
python msd_data_sample.py
```
### Output Files: R_train_sampled.npz, R_test_sampled.npz, df_content_final_sampled.csv, project_vars_sampled.pkl
### Step 2: User–Genre CF Model Training
```bash
python model_genre_cf.py
```
### Output Files: user_top_genres.pkl, model_genre_als.pkl
### Step 3: Hybrid Recommendation and Final Evaluation
```bash
python model_hybrid_final_final.py
```
### Result: Outputs the final performance metrics (Precision, Recall, Genre Similarity) and a qualitative analysis sample to the console.

## 📊 Evaluation Metrics
### Model performance is evaluated using the following metrics based on Top-K recommendations
### Precision@K: The fraction of recommended items in the top K that are relevant (found in the Test Set).
### Recall@K: The fraction of all relevant items in the Test Set that are successfully recommended in the top K.
### Genre Similarity (Cosine): Measures the cosine similarity between the average genre feature vector of the recommended songs and the average genre feature vector of the test set songs. (Assesses the genre consistency of recommendations).


