# Model Evaluation using Cross-Validation and Random Forest (Iris Dataset)

# Project Overview
This project demonstrates how to properly evaluate machine learning classification models using **cross-validation techniques** and improve performance using a **Random Forest classifier**. The **Iris dataset** is used as a benchmark multi-class classification dataset.

The project compares the performance of:
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree

Evaluation is done using **K-Fold** and **Stratified K-Fold Cross-Validation**, along with metrics such as accuracy, precision, recall, and F1-score.

---

# Objectives
- Apply **K-Fold** and **Stratified K-Fold Cross-Validation**
- Train and tune a **Random Forest** model
- Compare Random Forest with **SVM** and **Decision Tree**
- Evaluate model **generalization performance**
- Visualize performance using graphs and confusion matrix

---

# Dataset
The project uses the **Iris dataset**, which contains:

- 150 flower samples  
- 4 input features:
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- 3 classes:
  - Setosa  
  - Versicolor  
  - Virginica  

The dataset is balanced, with 50 samples per class.

---

# Technologies Used
- Python
- Scikit-learn
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Methodology

# Data Preprocessing
- Loaded Iris dataset from Scikit-learn
- Applied **Standardization** for scaling features (important for SVM)

# Cross-Validation
- **K-Fold (k=5)** for general performance evaluation  
- **Stratified K-Fold (k=5)** to maintain class balance

# Models Used
- Decision Tree  
- Support Vector Machine (SVM)  
- Random Forest  

# Hyperparameter Tuning
Used **GridSearchCV** to find optimal Random Forest parameters such as:
- Number of trees (`n_estimators`)
- Maximum depth (`max_depth`)
- Minimum samples per split

# Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

# Results Summary
Random Forest achieved the best performance due to ensemble learning, which reduces overfitting and improves generalization. Stratified K-Fold provided slightly more reliable results compared to standard K-Fold.

---

# How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

pip install -r requirements.txt
Run the notebook:

jupyter notebook
Open the .ipynb file and run all cells.

