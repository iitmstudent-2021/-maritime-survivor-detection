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

### 1. Data Files
The `survival-detection/` folder contains the competition data:
- `maritime_train.csv` - 714 passengers with survival outcomes (training data)
- `maritime_test.csv` - 181 passengers for prediction (test data)  
- `maritime_sample_submission.csv` - Example submission format

Note: The data has intentional Gaussian noise added to features and some columns are obfuscated to make it more challenging.

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Analysis
Open and run `data_exploration.ipynb` in Jupyter Notebook or VS Code

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
├── data_exploration.ipynb      # Main analysis notebook with EDA + modeling
├── survival-detection/         # Competition data folder
│   ├── maritime_train.csv      # Training data (714 passengers)
│   ├── maritime_test.csv       # Test data (181 passengers)
│   └── maritime_sample_submission.csv
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # You're reading it!
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

## 📝 Competition Rules
- Solo project (no teams allowed)
- Max 5 submissions per day - so choose wisely!
- Submit as CSV: PassengerName, Outcome
- Don't use external datasets or lookup the actual Titanic data

## 💭 What I Learned

The data is intentionally messy with Gaussian noise added everywhere. This actually made it more interesting - you can't just blindly trust your feature importance plots. 

A few things that worked:
- Gender is obviously the strongest predictor (women had way higher survival rates)
- Ticket class matters a lot - first class passengers had better odds
- Age groups work better than raw age values
- Family size features help, but be careful with the noise in those columns

What didn't work:
- Getting too aggressive with feature engineering (tried polynomial features, huge mistake)
- Trusting the public leaderboard too much - had several submissions that looked great but tanked in private LB
- Over-optimizing hyperparameters - sometimes simpler is better

## 🛠️ Technical Details

I ended up using a Random Forest with pretty standard hyperparameters. Tried XGBoost and ensemble methods but they didn't improve much on the validation set. The key was getting the preprocessing right - handling missing values carefully and creating a few good interaction features.

Current best: 81.7% on public leaderboard (fingers crossed for private!)

## ⚠️ Heads Up

The notebook has some experimental code at the bottom (rule-based overrides) that I was testing. It tries to manually override predictions for certain passenger types. Didn't help much but leaving it there in case anyone wants to try variations.

Also, if you're using this for your own submission, remember the data files are in the `survival-detection/` folder. Just run the notebook and it'll load them automatically.

---

Feel free to fork and improve! If you find a better approach, I'd love to hear about it.
