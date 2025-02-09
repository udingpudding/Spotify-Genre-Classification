# 🎵 Multi-Class Genre Classification with Neural Networks  

## 🚀 Project Goals  
This project aims to develop a multi-class classification model for predicting music genres with the following goals:  
1. **Great Generalizability** (Low Bias, Low Variance)  
2. **High Test AUROC**  
3. **Non-Complicated Feature Set** (Low Dimensionality)  
4. **Checks on Data Leakage to Ensure True Performance**  
5. **Compact and Fast Training Models**  

---

## 🔍 Model Choice  
After experimenting with multiple models—including tree-based ensemble methods (Random Forests, AdaBoost, XGBoost)—I chose **Neural Networks (NNs)** due to their **precise tunability and superior generalizability**.  

- Tree-based models exhibited **high variance** and **poor generalizability**.  
- NNs allowed fine-tuned control over **layers, neurons, learning rates, and epochs**, leading to **high AUROC** and **faster training times**.  

---

## 📌 Project Outline  

### **1️⃣ Data Cleaning & Preprocessing**  
- **Cleaning:** Addressed NaNs, incorrect values, and missing artist names.  
- **Imputation:**  
  - **Duration**: Used artist-level and global median values.  
  - **Tempo**: Trained an NN-based imputer (outperformed linear regression & tree-based models).  
- **Encoding & Standardization:**  
  - Ordinal encoding for categorical variables (Key, Mode, Genre).  
  - Standardization applied only to train set (avoiding leakage).  
- **Preventing Data Leakage:**  
  - **Shuffling before and after splitting**  
  - **Train-test separation before scaling and transformations**  
  - **Ensured PCA/LDA were fit only on the train set**  

---

### **2️⃣ Exploratory Data Analysis & Feature Selection**  
- **Histogram & Correlation Analysis** to assess distributions and outliers.  
- **Feature Importance Analysis** to select best predictors for classification.  
- **LDA Loadings Interpretation:**  
  - **Feature 1 (64%)** → *Danceability, Energy, Instrumentalness* (Interpreted as "Positivity")  
  - **Feature 2 (17%)** → *Popularity* (Interpreted as "Nicheness")  
  - **Feature 3 (11%)** → *Acousticness, Mode, Valence* (Interpreted as "Rhythm")  

---

### **3️⃣ Dimensionality Reduction & Clustering**  
- **Linear Methods**: PCA & LDA (LDA performed best in genre separation).  
- **Non-Linear Methods**: t-SNE (but had high overlap & long training times).  
- **Clustering:**  
  - **K-Means on LDA features** (Best result: **3 clusters**, silhouette score **0.35-0.44**).  

---

### **4️⃣ Model Building & Validation**  
Built **3 NN models** of varying complexity for different use cases:  

| Model        | AUROC (Macro) | Test Accuracy | Input Features | Hidden Layers | Parameters | Optimal Epochs | Train Time |
|-------------|--------------|---------------|----------------|---------------|------------|---------------|------------|
| **All Features** | 0.93         | 58%           | 13             | 2             | 755        | 125           | 34s        |
| **LDA 3D**  | 0.91         | 50%           | 3              | 1             | 220        | 100           | 24s        |
| **LDA 2D**  | 0.88         | 42%           | 2              | 1             | 205        | 150           | 17s        |
| **Gradient Boost Forest (Benchmark)** | 0.89 | 49% | 3 | N/A | N/A | 100 Estimators | 41s |

🏆 **Best Model:** **LDA 3D (3 Features, 1 Hidden Layer, AUROC = 0.91, Fastest Training)**  

#### 🔧 **Hyperparameter Tuning**  
- **Loss Function:** `nn.CrossEntropyLoss()`  
- **Optimizer:** Adam (adaptive learning rates)  
- **Learning Rates:** **0.1 or 0.01** (fine-tuned per model)  
- **Activation:** ReLU (fast convergence)  
- **Dropout:** **Not required** (regularization was not needed based on loss plots)  
- **Epochs:** **100-150**, stopping when **test loss increased over train loss** (indicating overfitting)  

---

### **5️⃣ Extra Credit: NLP for Track & Artist Names**  
To leverage text features, I vectorized track & artist names using **SpaCy embeddings** (300D each).  

#### 🏗 **Final Word-Feature Models**  
| Model | AUROC | Test Accuracy | Input Features | Hidden Layers | Parameters | Optimal Epochs | Train Time |
|-------|-------|--------------|----------------|---------------|------------|---------------|------------|
| **Word Features Only** | 0.88 | 60% | 600 | 4 | 146,885 | 40 | 21s |
| **All Features + Word Features** | 0.96 | 70% | 613 | 4 | 149,485 | 90 | 72s |

**📌 Key Takeaways:**  
✅ **Artist & track names improve genre classification**  
⚠️ **Potential leakage due to overlapping artists in train/test**  
🚨 **High complexity (600D vectors) → Need for dimensionality reduction**  

---

## 🔮 Going Forward  
- **Optimize hyperparameters further**  
- **Dimensionality reduction for word embeddings** (e.g., PCA, UMAP)  
- **Test smaller pre-trained SpaCy embeddings**  
- **Ensure train-test separation of artist names** to avoid leakage  

---


