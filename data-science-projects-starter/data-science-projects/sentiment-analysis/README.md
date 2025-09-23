
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

## ⚙️ Methods
1. **Data Cleaning**  
   - Removed URLs, mentions, hashtags, numbers, and special characters.  
   - Lowercased text.  
   - Optional: Tokenization and stopword removal.  

2. **Feature Engineering**  
   - Applied **TF-IDF vectorization** for text representation.  

3. **Modeling**  
   - Trained baseline models:  
     - Naive Bayes  
     - Logistic Regression  
   - Compared performance across classifiers.  

4. **Evaluation**  
   - Confusion Matrix  
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC curves  

---

## 📈 Results
- **Logistic Regression** performed best among baseline models.  
- Achieved strong accuracy and ROC-AUC scores on the test set.  

Example visualization (confusion matrix):  
![Confusion Matrix](plots/confusion_matrix.png)

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
