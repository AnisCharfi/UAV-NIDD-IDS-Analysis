# UAV-NIDD CRISP-DM: UAV Network Intrusion Detection System

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

This project implements a comprehensive **UAV Network Intrusion Detection System (IDS)** using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The system employs machine learning techniques to detect and classify cyber-attacks against Unmanned Aerial Vehicles (UAVs) in real-time, achieving a macro-averaged F1-score of **0.95** on the UAV-NIDD dataset.

## Project Objectives

- **Primary Goal**: Develop a high-performance ensemble machine learning model for UAV network intrusion detection
- **Target Performance**: Achieve >0.90 macro F1-score across all attack types
- **Real-world Application**: Create a deployable IDS for UAV-Ground Control Station (GCS) communications
- **Methodology**: Apply systematic CRISP-DM framework for reproducible cybersecurity research

## Dataset Information

### UAV-NIDD Dataset
- **Total Samples**: 860,643 network traffic records (after preprocessing)
- **Attack Types**: 11 distinct cyber-attack types + Normal traffic
- **Features**: 12 carefully selected wireless communication features (from 45 features)
- **Source**: Real UAV network traffic captured from industrial testbed
- **Hardware**: DJI Mavic Air, DJI Mini 3 Pro, PX4 Vision Dev Kit v1.5

### Attack Categories

**1. DDoS** - Distributed Denial of Service attacks (193,171 samples - 22.4%) <br>
**2. Brute Force**- Authentication brute force attacks (175,925 samples - 20.4%) <br>
**3. UDP Flooding** - UDP-based flooding attacks (152,775 samples - 17.8%) <br>
**4. MITM** - Man-in-the-Middle attacks (151,585 samples - 17.6%) <br>
**5. ICMP Flooding** - ICMP-based flooding attacks (47,544 samples - 5.5%) <br>
**6. Jamming** - Signal jamming attacks (41,600 samples - 4.8%) <br>
**7. Replay** - Traffic replay attacks (25,977 samples - 3.0%) <br>
**8. Fake Landing** - UAV landing command injection (24,050 samples - 2.8%) <br>
**9. Deauthentication** - WiFi deauthentication attacks (~18,000 samples - 2.1%) <br>
**10. Scanning** - Network reconnaissance attacks (~8,000 samples - 0.9%) <br>
**11. DoS** - Denial of Service attacks (~6,000 samples - 0.7%) <br>
**12. Reconnaissance** - Network reconnaissance attacks (~6 samples - <0.1%) <br>

<img width="1308" height="500" alt="Table 1 - UAV-NIDD Class Distribution" src="https://github.com/user-attachments/assets/6af3ff70-e9d7-4780-94c8-187e208d3c4e" />
 

##  Project Structure
In the end of the process, you should have a structure like this one below, so start with creating a folder called "workspace" and put the notebook inside it. <br>
<br>
```
/workspace/
├── UAV_NIDD_Crisp_Dm.ipynb          # Main notebook
├── Dataset-NIDD-with-category.xlsx   # Original dataset
├── Data_Understanding/               # Phase 1: EDA and analysis
├── Data_Preparation/                 # Phase 2: Cleaned datasets
│   └── final_cleaned_encoded_dataset.xlsx
├── Modeling/                         # Phase 3: ML experiments
│   ├── smote_baseline_results.xlsx
│   └── borderline_smote_baseline_results.xlsx
├── Evaluation/                       # Phase 4: Performance metrics
└── Deployment/                       # Phase 5: Final ensemble model
    └── final_best_of_best_ensemble.pkl
```

## Requirements

### Core Dependencies
```python
# Data Processing
pandas>=1.3.0
numpy>=1.21.0
openpyxl>=3.0.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
joblib>=1.1.0
scipy>=1.7.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/AnisCharfi/UAV-NIDD-IDS-Analysis.git
cd UAV-NIDD-CRISP-DM

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook UAV_NIDD_Crisp_Dm.ipynb
```

## Quick Start

### 1. Dataset Setup
Place the UAV-NIDD dataset file in the workspace directory:
```bash
/workspace/UAV-Case1-Label.csv
```
Afterwards, it will be converted (in the Data Understanding Phase) into Dataset-NIDD-with-category.xlsx

### 2. Running the Complete Pipeline
The notebook is organized into sequential cells following CRISP-DM phases:

```python
# Execute cells in order:
# 1. Data Understanding & EDA
# 2. Data Preparation & Feature Engineering  
# 3. SMOTE Baseline Modeling
# 4. Borderline-SMOTE Advanced Modeling
# 5. Hyperparameter Tuning
# 6. Ensemble Creation & Deployment
```

### 3. Model Deployment
The final trained ensemble model is saved as:
```python
final_model_path = "/workspace/Crisp-dm/Deployment/final_best_of_best_ensemble.pkl"
```

## Methodology

### CRISP-DM Implementation

#### Phase 1: Business Understanding
- **Problem Definition**: UAV networks face critical cybersecurity vulnerabilities
- **Success Criteria**: >0.90 macro F1-score across all attack types
- **Business Impact**: Real-time threat detection for UAV operations

#### Phase 2: Data Understanding
- **Dataset Analysis**: 860K+ samples, extreme class imbalance (32,195:1 ratio)
- **Feature Analysis**: 12 wireless communication features selected
- **Quality Assessment**: Missing data handling, outlier detection

#### Phase 3: Data Preparation
- **Class Balancing Strategy**: Two-stage resampling (Undersample → Oversample)
- **Feature Engineering**: Label encoding, standardization
- **Data Splitting**: 80/20 train-test split with stratification

#### Phase 4: Modeling
- **Algorithm Selection**: 7 diverse ML algorithms evaluated
- **Resampling Comparison**: SMOTE vs. Borderline-SMOTE
- **Cross-Validation**: Manual 5-fold StratifiedKFold implementation

#### Phase 5: Evaluation
- **Primary Metric**: Macro-averaged F1-score
- **Performance Analysis**: Per-class precision, recall, F1-score
- **Model Selection**: Top 3 performers identified

#### Phase 6: Deployment
- **Ensemble Design**: VotingClassifier with hard voting
- **Expert Models**: Random Forest, Decision Tree, XGBoost
- **Deployment Asset**: Serialized .pkl model file

## Key Results

### Final Performance Metrics
- **Macro F1-Score**: 0.95 (20% hold-out test set)
- **Individual Model Performance**:
  - Random Forest: 0.9640
  - Decision Tree: 0.9638  
  - XGBoost: 0.9634

### Resampling Strategy Comparison
| Method | Random Forest F1 | Decision Tree F1 | XGBoost F1 |
|--------|------------------|------------------|------------|
| SMOTE | 0.9538 | 0.9536 | 0.9534 |
| Borderline-SMOTE | 0.9640 | 0.9638 | 0.9634 |

### Attack Type Performance
- **Best Performance**: DDoS (Perfect recall: 1.00, Precision: 0.95)
- **Challenging Class**: DoS (F1-score: 0.52, limited training samples)
- **Overall Strength**: Excellent performance across MITM, Evil Twin, Replay attacks

## Technical Highlights

### Advanced Resampling Pipeline
```python
# Two-stage resampling strategy
1. RandomUnderSampler: Cap majority classes at 50,000 samples
2. BorderlineSMOTE: Generate synthetic minorities on decision boundaries
3. StandardScaler: Normalize features post-resampling
```

### Ensemble Architecture
```python
VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('dt', DecisionTreeClassifier()), 
        ('xgb', XGBClassifier())
    ],
    voting='hard'  # Majority vote decision
)
```

### Cross-Validation Framework
- **Manual Implementation**: Overcomes algorithmic instability issues
- **Label Consistency**: Robust LabelEncoder within each fold
- **Pipeline Control**: Explicit scaling → resampling → training sequence

## Performance Analysis

### Strengths
- **High Overall Performance**: 95% macro F1-score demonstrates excellent generalization
- **Robust Across Attack Types**: Consistent performance on diverse threat categories
- **Real-world Applicability**: Trained on authentic UAV network traffic
- **Interpretable Decisions**: Decision Tree component provides explainable results

### Limitations
- **DoS Attack Detection**: Lower precision (0.45) due to limited training samples
- **Computational Resources**: Ensemble requires more processing power than single models

## Advanced Usage

### Custom Feature Selection
```python
# Modify feature set in Data Preparation phase
feature_columns = [
    'frame.len', 'radiotap.channel.flags.cck', 'wlan.seq',
    'radiotap.dbm_antsignal', 'radiotap.rxflags',
    'wlan.rsn.capabilities.mfpc', 'radiotap.length',
    'wlan.fcs.bad_checksum', 'wlan.tag', 'wlan_radio.frequency',
    'wlan_radio.phy', 'udp.srcport'
]
```

### Hyperparameter Tuning
```python
# Enable hyperparameter optimization (optional)
# Note: Default parameters proved optimal in experiments
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

### Model Evaluation
```python
# Load and evaluate the final ensemble
import joblib
model = joblib.load('Deployment/final_best_of_best_ensemble.pkl')

# Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

##  Educational Value

This project serves as a comprehensive example of:
- **CRISP-DM Methodology**: Complete implementation of industry-standard data mining process
- **Imbalanced Learning**: Advanced techniques for handling severe class imbalance
- **Ensemble Methods**: Practical application of voting classifiers
- **Cybersecurity ML**: Domain-specific challenges and solutions
- **Reproducible Research**: Systematic experimental design and validation

##  References

1. UAV-NIDD Dataset: "A Dynamic Dataset for Cybersecurity and Intrusion Detection in UAV Networks"
link : https://ieeexplore.ieee.org/abstract/document/10937270?casa_token=ylzQplvW2O8AAAAA:cnrzom_WyvsHGFQiqufPbVfO_tl38uFWbWtHT1MUFKXgZTj64rNhp7MHeZKVdWCmz2kcpMiDvrWi

##  Contact

For questions, issues, or collaboration opportunities:
- **Email**: anischarfi2001@gmail.com
- **GitHub**: https://github.com/AnisCharfi/
- **LinkedIn**: https://www.linkedin.com/in/charfi-anis/

##  Acknowledgments

- UAV-NIDD dataset creators for providing comprehensive UAV network traffic data
- Open-source community for machine learning libraries and tools
- CRISP-DM framework developers for systematic data mining methodology
