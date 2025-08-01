# 📰 Fake News Detection in Arabic

This project implements a complete pipeline to detect fake news in **Arabic texts** using classical machine learning and deep learning models. It aims to provide an accurate and efficient solution for identifying misinformation in Arabic content.

---

## Pipeline Overview

1. **Load Dataset**  
   - Import Arabic news articles from a CSV file.

2. **Preprocess Text**  
   - Clean raw Arabic text using a custom `clean_arabic_text()` function.  
   - This includes removing punctuation, links, diacritics (tashkeel), and non-Arabic characters.

3. **Feature Extraction**  
   - Transform the cleaned text into numerical format using **TF-IDF vectorization**.

4. **Model Training**  
   - Train and compare multiple models:
     - `BERT` (Arabic pretrained transformer)
     - `SVC` (Support Vector Classifier)
     - `Random Forest`
     - `Logistic Regression`
     - `XGBoost`

5. **Model Evaluation**  
   - Evaluate model performance using:
     - ROC Curve  
     - Precision-Recall Curve  
     - AUC Score  
     - Accuracy, Precision, Recall, F1 Score

6. **Model Saving**  
   - Save the best-performing model and its corresponding vectorizer using `pickle`.

7. **Make Predictions**  
   - Load the saved model/vectorizer to classify new, unseen Arabic news articles as **Real** or **Fake**.

8. **Display Results**  
   - Output the predicted label with a confidence score for each input.

---

## 🛠️ Tech Stack

- Python (Pandas, scikit-learn, XGBoost, matplotlib, seaborn)
- TF-IDF for feature engineering
- BERT for contextual understanding
- Pickle for model serialization

---

## 📌 Future Improvements

- Integrate deep learning architectures like LSTM/CNN
- Deploy as a web app using Flask or FastAPI
- Support multilingual fake news detection
