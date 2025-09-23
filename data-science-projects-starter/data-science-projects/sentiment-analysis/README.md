# Sentiment Analysis on Twitter Data (Sentiment140 Dataset)

This project demonstrates sentiment classification on Twitter text using the **Sentiment140 dataset** (1.6M tweets).  
It is part of my portfolio to showcase skills in **data preprocessing, feature engineering, and machine learning model evaluation**.

---

## 🔹 Pipeline Overview
1. **Load Data**  
   - Auto-detect Kaggle dataset (Sentiment140)  
   - Fallback: load from local `data/sentiment.csv` (with columns `text`, `label`)

2. **Preprocessing & Cleaning**  
   - Remove URLs, @user mentions, hashtags  
   - Strip punctuation and extra whitespace  
   - Convert text to lowercase

3. **Dataset Split**  
   - Train/validation split for model evaluation  

4. **Feature Engineering**  
   - Apply **TF-IDF vectorization** (unigrams + bigrams)  

5. **Model Training**  
   - Naive Bayes  
   - Logistic Regression  

6. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix (saved to `plots/confusion_matrix.png`)  

7. **Model Persistence**  
   - Save best-performing model and TF-IDF vectorizer for reuse  

---

## 📊 Dataset
- **Name:** Sentiment140  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- **Size:** 1.6M tweets labeled as *positive (1)* or *negative (0)*  
- **Features:**
  - Tweet ID  
  - Date  
  - Query  
  - User  
  - Tweet text (cleaned for training)

📌 Note: The full dataset is too large to upload to GitHub.  
Only a **sample file (`data/sample.csv`)** is included for demonstration.  
For complete experiments, please download the dataset directly from Kaggle.

---

## 📈 Results
- **Logistic Regression** performed best among baseline models.  
- Achieved strong accuracy and ROC-AUC scores on the validation set.  

Example visualization (confusion matrix):  
<img width="657" height="492" alt="Screenshot 2025-09-23 163453" src="https://github.com/user-attachments/assets/6d71506a-98b6-4fb1-830d-8b97c7517222" />

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
