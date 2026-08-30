# AI_ML_Internship_Project
# Industrial AI/ML Internship – Quality Prediction in a Mining Process

**Industrial Internship Program | UniConverge Technologies Pvt. Ltd. (UCT)**
**Facilitated by upskill Campus / The IoT Academy | Duration: 6 Weeks**

This repository contains my internship report and project work completed during a **six-week industrial internship** focused on **Machine Learning for quality prediction in the mining and mineral-processing domain**.

The project, titled **“Quality Prediction in a Mining Process”**, focuses on developing a machine-learning-based **soft sensor** to predict the **percentage of Silica Concentrate (% Silica Concentrate)** in the final iron ore concentrate of a **froth flotation plant**. The project uses real industrial process data and explores data preprocessing, time-series analysis, feature engineering, regression modelling, model evaluation, and a future real-time deployment concept.

---

## 👤 Student Details

| Field                  | Details                                        |
| ---------------------- | ---------------------------------------------- |
| **Name**               | Bhupesh Singh Mehta                            |
| **Program**            | B.Tech. Computer Science & Engineering – AI/ML |
| **Internship Domain**  | Industrial AI / Machine Learning               |
| **Project**            | Quality Prediction in a Mining Process         |
| **Duration**           | 6 Weeks                                        |
| **Industrial Partner** | UniConverge Technologies Pvt. Ltd. (UCT)       |
| **Facilitated By**     | upskill Campus / The IoT Academy               |

---

## 📌 About the Internship

The internship was designed to bridge academic knowledge of **Artificial Intelligence and Machine Learning** with a practical industrial problem.

The project addressed the challenge of predicting silica impurity in iron ore concentrate before the final laboratory measurement becomes available. Since laboratory quality measurements are delayed, a machine-learning-based **soft sensor** can provide an earlier estimate of product quality using continuously available process measurements.

The internship followed the complete lifecycle of an industrial data-science project:

**Understand → Prepare → Explore → Engineer → Model → Validate → Communicate**

The project also provided practical exposure to industrial time-series data, data quality, temporal alignment, feature engineering, regression modelling, model evaluation, data leakage prevention, technical documentation, and AI solution presentation.

---

## 🎯 Project Objectives

The major objectives of the internship project were:

* Understand the industrial **iron ore froth flotation process**.
* Analyze real-world industrial time-series data.
* Predict **% Silica Concentrate** using Machine Learning.
* Investigate the feasibility of **minute-level quality prediction**.
* Study different **forecasting horizons** for silica concentration.
* Compare models with and without **% Iron Concentrate** as an input feature.
* Identify the process variables that contribute most to silica prediction.
* Compare regression models using **MAE, RMSE, and R²**.
* Study feature importance and model behaviour.
* Develop a leakage-safe and time-aware machine-learning workflow.
* Design a future architecture for deploying the prediction system through an industrial IoT/dashboard environment.

---

## ⛏️ Project: Quality Prediction in a Mining Process

### Problem Overview

Iron ore flotation plants generate large amounts of process data from sensors and control systems. However, the final product-quality measurement, particularly silica concentration, is obtained through laboratory analysis and is therefore delayed.

The project investigates whether Machine Learning can act as a **soft sensor** and estimate silica concentration earlier using available process measurements.

The target variable is:

**% Silica Concentrate**

The system uses process variables such as:

* % Iron Feed
* % Silica Feed
* Starch Flow
* Amina Flow
* Ore Pulp Flow
* Ore Pulp pH
* Ore Pulp Density
* Flotation Column Air Flow
* Flotation Column Levels
* % Iron Concentrate
* Other process measurements

The dataset contains approximately **737,453 records and 24 columns**, covering the period from **March to September 2017**. Some process variables are sampled every 20 seconds, while the final laboratory quality measurements are reported hourly.

---

## 🧠 Machine Learning Approach

The project treats the problem as a **time-series regression / industrial soft-sensor problem** rather than ordinary static regression.

The proposed workflow consists of:

1. **Data Acquisition**

   * Load industrial flotation-plant data.
   * Understand the available process and quality variables.

2. **Data Preprocessing**

   * Convert date/time values into proper datetime format.
   * Handle numeric formatting.
   * Check missing values and duplicates.
   * Validate data types and data quality.

3. **Exploratory Data Analysis**

   * Analyze distributions.
   * Study process trends.
   * Examine correlations.
   * Investigate relationships between process variables and silica concentration.

4. **Time Alignment**

   * Synchronize process measurements with laboratory quality measurements.
   * Resample data for different prediction frequencies.
   * Account for laboratory measurement delays.

5. **Feature Engineering**

   * Create lag features.
   * Generate rolling statistics.
   * Use only information available before the prediction timestamp.
   * Create different feature configurations.

6. **Model Development**

   * Mean prediction baseline
   * Linear Regression
   * Decision Tree
   * Random Forest
   * Gradient Boosting
   * XGBoost
   * LSTM/sequence models considered as future extensions

7. **Model Evaluation**

   * MAE – Mean Absolute Error
   * RMSE – Root Mean Squared Error
   * R² – Coefficient of Determination
   * Residual analysis
   * Feature importance

8. **Validation**

   * Chronological train/validation/test split.
   * Leakage auditing.
   * Forecast-horizon testing.
   * Robustness testing.

The proposed architecture separates **data engineering, modelling, evaluation, and monitoring**, allowing the machine-learning algorithm to be changed without redesigning the complete system.

---

## 🛠️ Technologies & Tools

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `Machine Learning` · `Time-Series Analysis` · `Regression` · `Feature Engineering` · `Industrial AI`

### Key Concepts

* Machine Learning
* Regression
* Time-Series Analysis
* Soft Sensors
* Data Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Lag Features
* Rolling Statistics
* Random Forest
* Gradient Boosting
* XGBoost
* LSTM
* Model Evaluation
* Data Leakage Prevention
* Industrial IoT

---

## 🗓️ Six-Week Internship Journey

| Week       | Focus Area                           | Major Activities                                                                                   |
| ---------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Week 1** | Orientation & Research               | Understanding mining/flotation processes, project requirements, research and data-leakage risks    |
| **Week 2** | Data Preparation & EDA               | Dataset analysis, cleaning, datetime conversion, missing values, correlations and trends           |
| **Week 3** | Time Alignment & Feature Engineering | Resampling, temporal alignment, lag features, rolling statistics and feature configurations        |
| **Week 4** | Machine Learning                     | Baseline regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost                   |
| **Week 5** | Model Tuning & Forecasting           | Hyperparameter tuning, forecast-horizon experiments and comparison with/without % Iron Concentrate |
| **Week 6** | Validation & Documentation           | Final architecture, validation strategy, deployment concept, documentation and presentation        |

The internship progressed from understanding the industrial problem through data preparation, feature engineering, modelling, evaluation, and final documentation.

---

## 🏗️ Proposed System Architecture

The proposed system follows this pipeline:

```text
Raw Industrial Data
        ↓
Data Preprocessing
        ↓
Timestamp & Numeric Standardization
        ↓
Time Synchronization / Resampling
        ↓
Lag & Rolling Feature Engineering
        ↓
ML Preprocessing Pipeline
        ↓
Regression Model
        ↓
Predicted % Silica Concentrate
        ↓
Evaluation & Monitoring
        ↓
Industrial Dashboard / IoT Platform
```

The system is designed to transform raw process measurements into an early prediction of silica concentration and eventually expose the prediction through an industrial monitoring or IoT application.

---

## 📊 Model Evaluation Strategy

The project emphasizes **time-aware and leakage-safe evaluation**.

The evaluation process includes:

* Chronological train/validation/test splitting
* Prediction timestamp definition
* Information cut-off enforcement
* Leakage auditing
* Forecast-horizon analysis
* Missing-value robustness testing
* Model comparison
* Residual analysis

The primary evaluation metrics are:

| Metric   | Purpose                                                           |
| -------- | ----------------------------------------------------------------- |
| **MAE**  | Measures average absolute prediction error                        |
| **RMSE** | Penalizes larger prediction errors more strongly                  |
| **R²**   | Measures the proportion of target variance explained by the model |

The report intentionally does **not claim an experimental test score**, because a runnable copy of the complete 737,453-row dataset was not included in the supplied materials. Published/reference results are treated separately from independently executed experiments.

---

## 💡 Key Learning Outcomes

This internship strengthened my understanding of both technical and industrial aspects of Machine Learning.

### Technical Skills

* Python programming
* Pandas and NumPy
* Data cleaning and transformation
* Datetime processing
* Time-series resampling
* Feature engineering
* Regression modelling
* Hyperparameter tuning
* Model evaluation
* Feature importance
* Data leakage prevention
* Industrial data analysis

### Professional Skills

* Understanding real-world engineering problems
* Research paper analysis
* Technical documentation
* Project presentation
* Problem-solving
* Communicating AI/ML solutions to an engineering audience

A major takeaway from the internship was that industrial Machine Learning is not only about selecting the best algorithm. **Data quality, temporal alignment, feature availability, validation strategy and deployment constraints are equally important.**

---

## 🚀 Future Scope

The project can be extended into a complete real-time industrial AI system by:

* Training the complete minute-level soft sensor using the full dataset.
* Performing systematic multi-hour forecasting experiments.
* Implementing LSTM, GRU, Temporal Convolution and Transformer-based models.
* Comparing deep-learning models with XGBoost.
* Adding explainability using SHAP or permutation importance.
* Implementing concept-drift monitoring.
* Creating a Streamlit/web dashboard.
* Connecting the prediction service with an industrial IoT platform.
* Adding prediction intervals and uncertainty estimation.
* Testing robustness against sensor failures and abnormal operating conditions.
* Periodically retraining the model using new laboratory measurements.

---

## 📁 Repository Contents

```text
├── AI_Internship_Report_Quality_Prediction_Mining_Process.pdf
└── README.md
```

Additional project notebooks, datasets, source code, presentation files, or diagrams can be added as the project repository is expanded.

---

## 🏆 Internship Outcome

The internship provided practical experience in applying **Artificial Intelligence and Machine Learning to an industrial mining problem**.

The project demonstrated how delayed laboratory measurements can potentially be complemented by an ML-based soft sensor, enabling earlier access to product-quality information.

Overall, the internship helped connect academic AI/ML concepts with **real-world industrial data, time-series modelling, predictive quality monitoring, and future Industrial IoT deployment**.

---

## 📚 References

1. Kaggle – *Quality Prediction in a Mining Process*
2. Moraes et al. (2018) – *Soft Sensor: Traditional Machine Learning or Deep Learning?*
3. Pu, Szmigiel & Apel (2020) – *Purities prediction in a manufacturing froth flotation plant: the deep learning techniques*
4. Literature on *FlotationNet* and deep-learning approaches for froth flotation prediction
5. Public project benchmarks on Random Forest and Gradient Boosting for the mining-process dataset

---

## 🙏 Acknowledgement

I would like to express my sincere gratitude to **UniConverge Technologies Pvt. Ltd. (UCT)**, **upskill Campus / The IoT Academy**, my mentors, coordinators, and academic faculty for providing guidance and support throughout this internship.

This internship gave me an opportunity to explore the practical application of Artificial Intelligence and Machine Learning in an industrial environment and significantly strengthened my technical and professional skills.

---

**Bhupesh Singh Mehta**
**B.Tech. Computer Science & Engineering – AI/ML**
