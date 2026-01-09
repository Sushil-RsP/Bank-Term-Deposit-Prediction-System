
# Bank Term Deposit Prediction

## Project Overview
This project analyzes a bank marketing dataset to predict whether clients will subscribe to a term deposit. The analysis includes comprehensive exploratory data analysis (EDA), feature engineering, and comparison of multiple machine learning classification models.

## Dataset
- **Source**: `bank.csv` (semicolon-delimited)
- **Target Variable**: `y` (subscription status - yes/no)
- **Features**: 20+ attributes including demographic, financial, and campaign-related information

### Key Features
- **Demographic**: age, job, marital status, education
- **Financial**: default status, housing loan, personal loan, balance
- **Campaign**: contact method, month, day of week, duration, number of contacts
- **Previous Campaign**: days since last contact, previous contacts, previous outcome

## Key Findings

### Data Insights
- **Class Imbalance**: Highly imbalanced dataset with ~88.7% non-subscribers and ~11.3% subscribers
- **Demographics**: 
  - Most clients work as admin, blue-collar, or technicians
  - Majority are married
  - Most have university degree or high school education
- **Campaign Patterns**: 
  - Primary contact method: phone
  - Peak contact months: May, July, August
  - Contacts distributed evenly across weekdays
- **Financial Status**: 
  - Very few clients have credit in default
  - Housing loans nearly evenly split
  - Most clients don't have personal loans

## Methodology

### 1. Data Loading & Exploration
- Load data with proper delimiter handling
- Check for missing values (none found)
- Analyze data types and structure
- Calculate unique values for categorical features

### 2. Exploratory Data Analysis (EDA)
Comprehensive visualization of all categorical features:
- Job distribution
- Marital status
- Education levels
- Default, housing, and personal loan status
- Contact methods
- Campaign timing (month, day of week)
- Previous campaign outcomes
- Target variable distribution

### 3. Data Preprocessing
- **Label Encoding**: Transform categorical variables to numerical format
- **Feature Scaling**: StandardScaler for normalization
- **Train-Test Split**: 80% training, 20% testing

### 4. Feature Importance Analysis
- Random Forest Classifier for feature importance ranking
- Comparison of balanced vs unbalanced models
- Visualization of top 15 most important features

### 5. Model Development
Three classification algorithms implemented and evaluated:

#### Logistic Regression
- Configuration: `class_weight='balanced'` to handle imbalanced data
- Linear probabilistic classifier

#### Naive Bayes (GaussianNB)
- Probabilistic classifier based on Bayes' theorem
- Fast training and prediction

#### K-Nearest Neighbors (KNN)
- Configuration: `n_neighbors=5`
- Instance-based learning algorithm

### 6. Model Evaluation
Each model evaluated using:
- **Accuracy Score**: Overall prediction correctness
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, Recall, F1-Score for each class

## Technical Stack

### Libraries
```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Data visualization
seaborn         # Statistical data visualization
scikit-learn    # Machine learning algorithms
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Running the Analysis
1. Ensure `bank.csv` is available in `C:\Users\LENOVO\Downloads\`
2. Open `Sushil_Chavan.ipynb` in Jupyter Notebook or VS Code
3. Run cells sequentially from top to bottom
4. Review visualizations and model performance metrics

### Expected Output
- Distribution plots for all categorical features
- Feature importance rankings and visualizations
- Model performance metrics (accuracy, confusion matrix, classification report)

## Project Structure
```
.
├── Sushil_Chavan.ipynb    # Main analysis notebook
├── README.md              # Project documentation (this file)
└── bank.csv              # Dataset (in Downloads folder)
```

## Results Summary
The notebook demonstrates:
- Thorough exploratory data analysis with 25+ visualizations
- Feature importance analysis using Random Forest
- Comparison of three different classification approaches
- Handling of imbalanced datasets using class weights
- Complete model evaluation pipeline

## Future Enhancements
- **Imbalance Handling**: Implement SMOTE or undersampling techniques
- **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV
- **Advanced Models**: XGBoost, LightGBM, CatBoost
- **Feature Engineering**: Create interaction features and polynomial features
- **Cross-Validation**: K-fold cross-validation for robust evaluation
- **ROC-AUC Analysis**: Threshold optimization and ROC curve visualization
- **Ensemble Methods**: Voting classifier or stacking
- **Feature Selection**: Recursive Feature Elimination (RFE)

## Author
Sushil Chavan

## License
This project is for educational and analytical purposes.

---
*Last Updated: January 2026*
