# Portuguese Student Grade Analysis & Prediction

A comprehensive machine learning project analyzing Portuguese high school student performance and predicting final grades using academic, demographic, and social factors.

## Project Overview

This project performs exploratory data analysis (EDA) and builds predictive models to forecast Portuguese high school students' final grades (G3) based on various personal, academic, and social characteristics.

**Dataset**: Portuguese secondary school student performance data from UCI ML Repository

## Repository Structure

```
├── StudentGradeAnalysis.ipynb    # Main Jupyter notebook with complete analysis
├── StudentGradeAnalysis.pdf      # PDF export of the notebook
├── StudentGrades.csv            # Student performance dataset
└── requirements.txt                    # Project drequirements
```

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
cufflinks>=0.17.3
plotly>=5.0.0
jupyter>=1.0.0
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn cufflinks plotly jupyter
```

## Objectives

- Primary Goal: Predict final grades (G3) of Portuguese high school students
- Secondary Goals: 
  - Identify key factors affecting academic performance
  - Understand relationships between demographic/social factors and grades
  - Compare multiple regression algorithms for optimal prediction

## Key Findings

### Demographic Insights
- Gender Distribution: Nearly equal (female vs male students)
- Age Range: Primarily 15-19 years old
- Location: 77.72% urban, 22.28% rural students
- Family Structure: 71.14% from families with >3 members

### Performance Factors
1. Previous Failures: Strong negative correlation with final grades
2. Family Education: Combined parent education positively impacts performance
3. Higher Education Aspirations: Students wanting higher education score better
4. Social Life: Students who go out frequently tend to score lower
5. Relationships: Students without romantic relationships perform better

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Data Quality: No missing values in 395 student records
- Visualization Tools: Seaborn, Matplotlib, Cufflinks
- Analysis Types: Distribution plots, correlation heatmaps, box plots, swarm plots

### 2. Data Preprocessing
- Label Encoding: Categorical variables converted to numerical format
- Feature Engineering: Created family education composite feature
- Feature Selection: Kept top 9 most correlated features with target variable
- Data Split: 75% training, 25% testing

### 3. Machine Learning Models

| Model | Type | Purpose |
|-------|------|---------|
| Linear Regression | Baseline | Simple linear relationship modeling |
| ElasticNet Regression | Regularized | Handles multicollinearity |
| Random Forest | Ensemble | Non-linear patterns, feature importance |
| Extra Trees | Ensemble | Reduced overfitting |
| Support Vector Machine | Kernel-based | Non-linear decision boundaries |
| Gradient Boosting | Boosting | Sequential learning |

### 4. Evaluation Metrics
- MAE (Mean Absolute Error): Average prediction error magnitude
- RMSE (Root Mean Square Error): Penalizes larger errors more heavily

## Results

### Model Performance
**Best Performing Model**: Linear Regression
- Achieved lowest MAE and RMSE scores
- Outperformed complex ensemble methods
- Suggests linear relationships dominate the data

### Feature Importance Rankings
1. failures (previous academic failures)
2. higher (desire for higher education)
3. father_education (father's education level)
4. mother_education (mother's education level)
5. goes_out (frequency of going out with friends)
6. romantic (romantic relationship status)
7. reason (reason for choosing school)

## Usage

1. Clone the repository
```bash
git clone https://github.com/yashk1103/Student-Grade-Analysis.git
cd student-grade-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook
```bash
jupyter notebook StudentGradeAnalysis.ipynb
```

## Analysis Sections
- Data Loading & Exploration
- Visualization & EDA
- Data Preprocessing
- Model Training & Evaluation

## Key Visualizations
- Correlation Heatmaps: Feature relationships with final grades
- Distribution Plots: Grade distributions across different demographics
- Box Plots: Performance comparisons by categorical variables
- Swarm Plots: Detailed grade distributions by key factors
- Model Comparison Charts: MAE and RMSE performance across algorithms

## Technologies Used
- Python 3.x
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib & Seaborn - Data visualization
- Scikit-learn - Machine learning algorithms
- Cufflinks - Interactive plotting
- Jupyter Notebook - Development environment

## Key Insights
1. Academic History Matters: Previous failures are the strongest predictor
2. Family Education Impact: Combined parent education significantly affects performance
3. Motivation Factor: Students aspiring for higher education perform better
4. Social Balance: Excessive social activities negatively impact grades
5. Relationship Effects: Romantic relationships may distract from academics
6. Model Simplicity: Linear models outperformed complex ensemble methods

## Model Interpretation
The analysis reveals that student academic success is primarily driven by:
- Past academic performance (failures)
- Educational environment (family education)
- Personal motivation (higher education aspirations)
- Life balance (social activities, relationships)

## Future Enhancements
- Feature Engineering: Create more sophisticated composite features
- Time Series Analysis: Analyze grade progression over time (G1 → G2 → G3)
- Deep Learning: Implement neural networks for pattern recognition
- Cross-validation: Implement k-fold cross-validation for robust evaluation
- Hyperparameter Tuning: Optimize model parameters using grid search

## Acknowledgments
- Dataset Source: UCI Machine Learning Repository
- Original Research: P. Cortez and A. Silva (2008) student performance prediction study
