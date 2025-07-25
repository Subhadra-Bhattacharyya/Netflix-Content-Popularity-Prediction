# 🎬 Netflix Content Popularity Prediction

Predicting the popularity of Netflix titles using machine learning and metadata analysis.

---

## 📌 Overview

This project aims to build a supervised machine learning model to predict whether a Netflix show or movie will be popular based on metadata like title description, genre, release year, and type (Movie/TV Show). This prediction can help improve recommendation systems and support data-driven decisions in content production and marketing.

---

## 🚀 Project Goals

- Analyze Netflix content data.
- Engineer features from both structured and unstructured fields (e.g., descriptions).
- Train machine learning models to classify content as "popular" or "not popular".
- Evaluate multiple models to identify the best-performing classifier.
- Visualize genre trends and description-based insights.

---

## 📂 Dataset

- **Source**: [Kaggle – Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Attributes Used**:
  - `title`, `description`, `type`, `release_year`, `listed_in`, `country`, `duration`
- **Target**:
  - `popularity` (manually labeled or derived)

---

## ⚙️ Methodology

1. **Data Cleaning**: Handled nulls, removed duplicates, and standardized formats.
2. **Text Preprocessing**:
   - Cleaned and tokenized `description` field.
   - Applied TF-IDF vectorization for feature extraction.
3. **Categorical Encoding**:
   - One-hot encoded fields like `type` and `genre`.
4. **Model Training**:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (best performance)
   - SVM
5. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score, ROC-AUC

---

## 🧪 Results

| Model                 | Accuracy | F1 Score |
|-----------------------|----------|----------|
| Logistic Regression   | 76%      | 74%      |
| Random Forest         | 83%      | 80%      |
| XGBoost Classifier    | 85%      | 83%      |
| Support Vector Machine| 78%      | 76%      |

✅ **XGBoost** emerged as the most reliable model for this binary classification task.

---

## 📈 Visualizations

- 📊 Genre-wise popularity trends
- ☁️ WordClouds for popular content descriptions
- 📉 Distribution plots for year, duration, etc.

---

## 💻 Technologies Used

| Category        | Tools / Libraries                          |
|-----------------|---------------------------------------------|
| Programming     | Python                                      |
| Data Handling   | Pandas, NumPy                               |
| Visualization   | Matplotlib, Seaborn, WordCloud              |
| ML Models       | Scikit-learn, XGBoost, RandomForest         |
| NLP             | TF-IDF Vectorizer (from Scikit-learn)       |
| IDE/Notebook    | Jupyter Notebook                            |

---

Install dependencies:
pip install -r requirements.txt

Run the Jupyter notebook:
jupyter notebook Netflix_Content_Popularity_prediction.ipynb

🔮 Future Work
1. Use BERT or transformer-based models for better description understanding.
2. Incorporate user behavior or rating data.
3. Build a Flask or Streamlit web app for live prediction.
4. Explore multi-class classification (e.g., Very Popular, Average, Unpopular).

📚 References
Netflix Dataset – Kaggle

Scikit-learn Documentation

XGBoost Paper

TF-IDF Guide – Sklearn

👨‍💻 Author
Subhadra Bhattacharyya
Data Science Enthusiast | ML Developer
Contact: subhadrabhattacharyya7@gmail.com
Feel free to reach out for collaboration or questions!


