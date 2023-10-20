# Text Classification using XGBoost
![image](https://github.com/kaciakk/Project-PJN/assets/95936444/55c6dc8d-55c5-4548-957a-e7850c5f18c6)


![image](https://github.com/kaciakk/Project-PJN/assets/95936444/3bab8d3d-4037-4334-8ec1-f3474c7cf225)

## Overview

This project demonstrates a text classification task using the XGBoost algorithm. The goal is to classify text data into two categories: duplicate and non-duplicate. We will utilize the Quora Question Pairs dataset, containing pairs of questions and labels that indicate whether the questions are duplicates. The project involves data preprocessing, feature extraction using TF-IDF, model training with XGBoost, and performance evaluation.

## Instructions
   0.5. **Test.csv download**:
   https://drive.google.com/drive/folders/1CUWfbf5Gc-f5aVtIcw_WL9GtPdPzQeGB?usp=sharing

   
Follow the instructions below to understand the project and run the code successfully:

1. **Data Preparation**:
   - The project starts with unpacking the dataset files from "train.csv.zip" and "test.csv.zip" using the `shutil` library.

2. **Data Loading**:
   - The `read_data` function reads the dataset and handles missing values. The cleaned dataset is stored in a Pandas DataFrame.

3. **Data Selection**:
   - The relevant columns (i.e., "question1," "question2," and "is_duplicate") are selected for analysis.

4. **Data Preprocessing**:
   - The `preprocess_data` function applies various text preprocessing steps to clean the text data. This includes converting to lowercase, removing contractions, hashtags, special characters, links, non-ASCII characters, punctuation, digits, and single letters, as well as eliminating extra spaces.

5. **Data Splitting**:
   - The dataset is split into training and development sets using `train_test_split`. The split ratio is 70% for training and 30% for development. Stratification ensures a balanced distribution of the "is_duplicate" labels.

6. **Feature Extraction**:
   - The TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is applied to convert the text data into numerical features. It is performed on the preprocessed text from both questions using `TfidfVectorizer`.

7. **Model Training**:
   - An XGBoost classifier is used for training on the TF-IDF features. The target variable is "is_duplicate." The trained model is stored in `model_xgb`.

8. **Model Evaluation**:
   - Model predictions are generated for the development set, and the accuracy score is computed using `accuracy_score`. A confusion matrix is created using `confusion_matrix` and visualized with `plot_confusion_matrix` from `mlxtend`.

## Technologies Used

The project employs the following technologies:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Regular Expressions (Regex)
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization

