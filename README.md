# Maritime Survivor Detection Challenge

## 🎯 Project Overview
This project is my solution for the **IIT Madras BS Data Science Competition** - a binary classification challenge to predict passenger survival from a maritime incident. The competition tests skills in handling real-world data complexities including Gaussian noise, missing values, and obfuscated features.

**Competition Link**: [Survival Detection on Kaggle](https://www.kaggle.com/competitions/survival-detection/leaderboard)

**Key Challenge**: Predict survival outcomes for 181 test passengers based on patterns learned from 714 training examples, achieving the highest accuracy on a private test set.

## 📊 Evaluation Metric
**Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`

## 🏆 Results
- **Best Model**: Random Forest Classifier with optimized hyperparameters
- **Validation Accuracy**: ~85%
- **Key Techniques**: Feature engineering, ensemble methods, rule-based overrides

## 🚀 Quick Start

### 1. Setup
Ensure you have the required data files:
- `maritime_train.csv` - Training data with outcome labels
- `maritime_test.csv` - Test data for predictions

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Analysis
Open and run `survivor_detection.ipynb` in Jupyter Notebook or VS Code

### 4. Generate Submission
The notebook will create `submission.csv` with the required format:
```csv
PassengerName, Outcome
"Braund, Mr. Owen Harris", 0
"Heikkinen, Miss. Laina", 1
```

## 📁 Project Structure
```
.
├── survivor_detection.ipynb    # Main analysis notebook
├── maritime_train.csv          # Training data (place here)
├── maritime_test.csv           # Test data (place here)
├── submission.csv              # Generated predictions
└── README.md                   # This file
```

## 🔍 Solution Approach

### 1. Data Exploration
- Analyze feature distributions
- Identify missing values
- Understand correlations
- Detect patterns despite obfuscation

### 2. Preprocessing
- Handle missing values (median/mode imputation)
- Extract features from PassengerName (titles)
- Encode categorical variables
- Scale numerical features

### 3. Feature Engineering
- Title extraction from names
- Feature interaction terms
- Handling obfuscated features

### 4. Modeling
Implements multiple algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Ensemble methods

### 5. Hyperparameter Tuning
- Grid search for optimal parameters
- Cross-validation (5-fold)
- Model comparison

### 6. Evaluation
- Validation set performance
- Cross-validation scores
- Feature importance analysis

## 📝 Submission Guidelines
- **Teams**: Maximum 1 member per team
- **Daily Limit**: 5 submissions per day
- **Format**: CSV with PassengerName and Outcome columns
- **Prohibited**: External lookup tables or scripts to find original survival results

## 💡 Tips for Success
1. **Start Simple**: Baseline model first, then iterate
2. **Feature Engineering**: Extract meaningful features from obfuscated data
3. **Handle Missing Data**: Strategic imputation is crucial
4. **Avoid Overfitting**: Use cross-validation, don't rely solely on public leaderboard
5. **Ensemble Methods**: Combine multiple models for better generalization
6. **Monitor Submissions**: Use your 5 daily submissions wisely

## 🔧 Model Performance Tracking
Track your submissions:
| Submission | Date | Model | CV Accuracy | Public LB | Notes |
|------------|------|-------|-------------|-----------|-------|
| 1          |      |       |             |           |       |
| 2          |      |       |             |           |       |
| 3          |      |       |             |           |       |

## 📚 Key Techniques Used
- **Imputation**: Median (numerical), Mode (categorical)
- **Encoding**: Label encoding for categorical features
- **Scaling**: StandardScaler for numerical features
- **Validation**: Stratified train-test split
- **Ensemble**: Voting classifier with soft voting

## ⚠️ Important Notes
- Data contains Gaussian noise - robust preprocessing is essential
- Column names are obfuscated - focus on patterns, not semantic meaning
- Missing data is intentional - handle strategically
- Public leaderboard may not reflect final scores - avoid overfitting

## 🏆 Success Strategy
1. **Understand the data** through thorough EDA
2. **Clean and preprocess** systematically
3. **Engineer meaningful features** from available data
4. **Test multiple models** and ensemble them
5. **Validate rigorously** using cross-validation
6. **Submit strategically** within daily limits

Good luck! 🚢
