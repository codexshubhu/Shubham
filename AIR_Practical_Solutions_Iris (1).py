"""
=============================================================================
 Wadia College of Engineering, Pune-01
 DEPARTMENT OF AUTOMATION AND ROBOTICS
 T.E (2019) Practical Exam - Artificial Intelligence in Robot
 COMPLETE SOLUTIONS USING IRIS DATASET
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMON IMPORTS (run once at the top)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "Iris.csv"   # change path if needed


# =============================================================================
# Q1 – Load dataset, display first 10 rows, shape, data types  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q1: Loading and Exploring the Dataset")
print("="*60)

df = pd.read_csv(DATASET_PATH)

print("\n--- First 10 Rows ---")
print(df.head(10))

print("\n--- Shape of Dataset ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Data Types of All Columns ---")
print(df.dtypes)

print("\n--- Basic Info ---")
print(df.info())


# =============================================================================
# Q2 – Four different visualizations  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q2: Data Visualizations")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset – Four Visualizations", fontsize=16)

# 1. Histogram – SepalLengthCm
axes[0, 0].hist(df['SepalLengthCm'], bins=15, color='steelblue', edgecolor='black')
axes[0, 0].set_title("Histogram – Sepal Length")
axes[0, 0].set_xlabel("Sepal Length (cm)")
axes[0, 0].set_ylabel("Frequency")
# Reveals: distribution shape; slightly right-skewed across species.

# 2. Box Plot – all numeric features
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df[numeric_cols].plot(kind='box', ax=axes[0, 1])
axes[0, 1].set_title("Box Plot – All Numeric Features")
axes[0, 1].set_ylabel("Value (cm)")
# Reveals: spread, median, and outliers in SepalWidthCm.

# 3. Scatter Plot – Sepal Length vs Petal Length
for species, grp in df.groupby('Species'):
    axes[1, 0].scatter(grp['SepalLengthCm'], grp['PetalLengthCm'], label=species, alpha=0.7)
axes[1, 0].set_title("Scatter – Sepal Length vs Petal Length")
axes[1, 0].set_xlabel("Sepal Length (cm)")
axes[1, 0].set_ylabel("Petal Length (cm)")
axes[1, 0].legend()
# Reveals: clear linear separation between Iris-setosa and others.

# 4. Bar Chart – Average Petal Width per Species
avg_petal = df.groupby('Species')['PetalWidthCm'].mean()
avg_petal.plot(kind='bar', ax=axes[1, 1], color=['salmon', 'lightgreen', 'skyblue'],
               edgecolor='black')
axes[1, 1].set_title("Bar Chart – Avg Petal Width per Species")
axes[1, 1].set_xlabel("Species")
axes[1, 1].set_ylabel("Avg Petal Width (cm)")
axes[1, 1].tick_params(axis='x', rotation=30)
# Reveals: Iris-virginica has largest petal width; setosa has smallest.

plt.tight_layout()
plt.savefig("q2_visualizations.png", dpi=150)
plt.show()
print("Plot saved as q2_visualizations.png")


# =============================================================================
# Q3 – Data Acquisition Process  (5 marks – Theory)
# =============================================================================
print("\n" + "="*60)
print("Q3: Data Acquisition Process (Theory)")
print("="*60)
theory_q3 = """
Data Acquisition is the FIRST step in any ML pipeline. It involves:

1. Identifying Data Sources
   - Kaggle datasets, UCI ML Repository, IoT sensors, APIs, web scraping,
     databases (SQL/NoSQL), or manual collection.

2. Data Collection Methods
   - Downloading CSV/Excel/JSON files
   - Querying databases (SQLAlchemy, psycopg2)
   - Calling REST APIs (requests library)
   - Web scraping (BeautifulSoup, Scrapy)
   - Real-time sensor data (MQTT, serial ports)

3. Data Storage
   - Raw data stored in CSV, Parquet, SQL tables, or cloud storage (S3).

4. Data Versioning
   - Tools like DVC (Data Version Control) track dataset versions.

5. Quality Check
   - Check for completeness, consistency, format, and relevance.

Example: In this exam, the Iris CSV was downloaded from Kaggle
         and loaded using pd.read_csv().
"""
print(theory_q3)


# =============================================================================
# Q4 – Handling Missing Values  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q4: Handling Missing Values")
print("="*60)

df_miss = df.copy()

# Artificially introduce missing values for demonstration
np.random.seed(42)
missing_idx = np.random.choice(df_miss.index, size=15, replace=False)
df_miss.loc[missing_idx[:8], 'SepalLengthCm'] = np.nan
df_miss.loc[missing_idx[8:], 'PetalWidthCm'] = np.nan

print("Missing values before handling:")
print(df_miss.isnull().sum())

# Strategy 1: Drop rows with missing values
df_dropped = df_miss.dropna()
print(f"\nStrategy 1 – Drop Rows: {len(df_miss)} → {len(df_dropped)} rows")

# Strategy 2: Mean/Median imputation
df_imputed = df_miss.copy()
df_imputed['SepalLengthCm'].fillna(df_imputed['SepalLengthCm'].mean(), inplace=True)
df_imputed['PetalWidthCm'].fillna(df_imputed['PetalWidthCm'].median(), inplace=True)
print("\nStrategy 2 – Mean/Median Imputation:")
print(df_imputed.isnull().sum())
print("SepalLengthCm filled with MEAN:", round(df_miss['SepalLengthCm'].mean(), 3))
print("PetalWidthCm filled with MEDIAN:", round(df_miss['PetalWidthCm'].median(), 3))


# =============================================================================
# Q5 – Outlier Detection & Removal using IQR  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q5: Outlier Detection and Removal (IQR Method)")
print("="*60)

df_clean = df.copy()
feature = 'SepalWidthCm'

Q1 = df_clean[feature].quantile(0.25)
Q3 = df_clean[feature].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"Lower Bound={lower_bound}, Upper Bound={upper_bound}")

outliers = df_clean[(df_clean[feature] < lower_bound) | (df_clean[feature] > upper_bound)]
print(f"Outliers detected: {len(outliers)}")

df_no_outliers = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
print(f"Rows before: {len(df_clean)}, After removing outliers: {len(df_no_outliers)}")

# Box plots before and after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.boxplot(df_clean[feature])
ax1.set_title("Before Outlier Removal")
ax1.set_ylabel(feature)

ax2.boxplot(df_no_outliers[feature])
ax2.set_title("After Outlier Removal (IQR)")
ax2.set_ylabel(feature)
plt.suptitle("Q5: Outlier Removal – SepalWidthCm")
plt.savefig("q5_outliers.png", dpi=150)
plt.show()
print("Comment: The box after removal shows no whisker points beyond ±1.5*IQR.")


# =============================================================================
# Q6 – Skewness & Transformations  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q6: Skewness and Transformations")
print("="*60)

from scipy import stats

numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("\nSkewness values:")
for col in numeric_cols:
    print(f"  {col}: {df[col].skew():.4f}")

# Use PetalLengthCm (skewed)
col_skewed = 'PetalLengthCm'
original = df[col_skewed]

log_transformed   = np.log1p(original)
sqrt_transformed  = np.sqrt(original)
boxcox_transformed, _ = stats.boxcox(original + 1)

print(f"\nOriginal skew         : {original.skew():.4f}")
print(f"Log Transform skew    : {log_transformed.skew():.4f}")
print(f"Sqrt Transform skew   : {sqrt_transformed.skew():.4f}")
print(f"Box-Cox Transform skew: {pd.Series(boxcox_transformed).skew():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Q6: Skewness Transformation – PetalLengthCm")

for ax, data, title in zip(
    axes.flatten(),
    [original, log_transformed, sqrt_transformed, pd.Series(boxcox_transformed)],
    ["Original", "Log Transform", "Sqrt Transform", "Box-Cox Transform"]
):
    ax.hist(data, bins=20, color='orchid', edgecolor='black')
    ax.set_title(f"{title} (skew={data.skew():.3f})")

plt.tight_layout()
plt.savefig("q6_skewness.png", dpi=150)
plt.show()


# =============================================================================
# Q7 – Handling Categorical Data  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q7: Handling Categorical Data")
print("="*60)

df_cat = df.copy()
print("Original 'Species' values:", df_cat['Species'].unique())

# Technique 1: Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cat['Species_LabelEncoded'] = le.fit_transform(df_cat['Species'])
print("\nLabel Encoding:")
print(df_cat[['Species', 'Species_LabelEncoded']].drop_duplicates())
print("Use when: ordinal data or tree-based models (no magnitude assumption).")

# Technique 2: One-Hot Encoding
df_ohe = pd.get_dummies(df_cat, columns=['Species'], prefix='species')
print("\nOne-Hot Encoding (first 3 rows):")
print(df_ohe[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].head(3))
print("Use when: nominal data + linear models to avoid ordinal bias.")


# =============================================================================
# Q8 – Heatmap, Top Features, Train-Test Split  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q8: Correlation Heatmap, Top Features, Train-Test Split")
print("="*60)

from sklearn.model_selection import train_test_split

df_enc = df.copy()
df_enc['Species'] = LabelEncoder().fit_transform(df_enc['Species'])
df_enc.drop(columns=['Id'], inplace=True)

# Heatmap
plt.figure(figsize=(8, 6))
corr = df_enc.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Q8: Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("q8_heatmap.png", dpi=150)
plt.show()

# Top 5 features correlated to 'Species'
print("\nCorrelation with target 'Species':")
print(corr['Species'].sort_values(ascending=False))

# Train-Test Split
X = df_enc.drop('Species', axis=1)
y = df_enc['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")


# =============================================================================
# Q9 – Principal Component Analysis (PCA)  (20 marks)
# =============================================================================
print("\n" + "="*60)
print("Q9: PCA – Dimensionality Reduction")
print("="*60)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Theory
theory_pca = """
AIM: Reduce the dimensionality of the Iris dataset while retaining maximum variance.

THEORY:
PCA transforms the original correlated features into a new set of uncorrelated
variables called Principal Components (PCs) ordered by variance explained.

Steps:
1. Standardize the data (zero mean, unit variance).
2. Compute the covariance matrix.
3. Compute eigenvalues and eigenvectors.
4. Sort eigenvectors by descending eigenvalue.
5. Project data onto top-k eigenvectors.

ADVANTAGES:
  - Removes multicollinearity.
  - Reduces overfitting.
  - Speeds up training.
  - Enables 2D/3D visualization of high-dimensional data.

DISADVANTAGES:
  - Loss of interpretability (PCs are linear combinations).
  - Assumes linear relationships.
  - Sensitive to outliers and feature scaling.
  - May discard features that are weak but important.
"""
print(theory_pca)

X_pca = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_labels = df['Species'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Plot
plt.figure(figsize=(8, 5))
for label in np.unique(y_labels):
    idx = y_labels == label
    plt.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1], label=label, alpha=0.8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Q9: PCA – 2D Projection of Iris Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("q9_pca.png", dpi=150)
plt.show()


# =============================================================================
# Q10 – Feature Engineering & StandardScaler  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q10: Feature Engineering & Standard Scaler")
print("="*60)

theory_fe = """
FEATURE ENGINEERING: The process of using domain knowledge to create, transform,
or select features that improve model performance.
  - Examples: polynomial features, interaction terms, binning, encoding.

STANDARD SCALER (Z-score normalization):
  Formula: z = (x - mean) / std
  - Transforms features to have mean=0 and std=1.
  - Essential for distance-based models (KNN, SVM) and gradient-based models.

WHY SCALE AFTER SPLITTING?
  - If scaler is fit on the full dataset (including test), test statistics
    'leak' into training — this is data leakage.
  - Correct: fit scaler on X_train ONLY, then transform both X_train and X_test.
"""
print(theory_fe)

from sklearn.preprocessing import StandardScaler

df_s = df.drop(columns=['Id'])
df_s['Species'] = LabelEncoder().fit_transform(df_s['Species'])
X_s = df_s.drop('Species', axis=1)
y_s = df_s['Species']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_s)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test_s)        # transform only on test

print("Before Scaling (train mean):", X_train_s.mean().values.round(3))
print("After Scaling  (train mean):", X_train_scaled.mean(axis=0).round(3))
print("After Scaling  (train std) :", X_train_scaled.std(axis=0).round(3))


# =============================================================================
# Q11 – Regression Model  (20 marks)
# =============================================================================
print("\n" + "="*60)
print("Q11: Linear Regression Model")
print("="*60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict PetalLengthCm from SepalLengthCm
X_reg = df[['SepalLengthCm']].values
y_reg = df['PetalLengthCm'].values

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

mae  = mean_absolute_error(y_te, y_pred)
mse  = mean_squared_error(y_te, y_pred)
r2   = r2_score(y_te, y_pred)

print(f"Coefficients : {model.coef_[0]:.4f}")
print(f"Intercept    : {model.intercept_:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"R²   : {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_te, y_te, color='steelblue', label='Actual', alpha=0.8)
plt.plot(X_te, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Q11: Linear Regression – Sepal Length vs Petal Length")
plt.legend()
plt.tight_layout()
plt.savefig("q11_regression.png", dpi=150)
plt.show()


# =============================================================================
# Q12 – Regression vs Classification  (10 marks – Theory)
# =============================================================================
print("\n" + "="*60)
print("Q12: Regression vs Classification (Theory)")
print("="*60)
theory_q12 = """
REGRESSION MODEL:
  - Predicts a CONTINUOUS output (e.g., house price, temperature).
  - Evaluation Parameters:
      1. MAE  (Mean Absolute Error) – Average absolute difference.
      2. R²   (R-Squared) – Proportion of variance explained (0 to 1).

CLASSIFICATION MODEL:
  - Predicts a DISCRETE class label (e.g., spam/not-spam, species).
  - Evaluation Parameters:
      1. Accuracy = (TP + TN) / Total
      2. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

WHEN TO CHOOSE REGRESSION OVER CLASSIFICATION?
  - When the target variable is continuous and ordered (price, length, weight).
  - Example: Predicting petal length from sepal length in Iris dataset.
  - Use Classification when target is a discrete label (e.g., Iris species).
"""
print(theory_q12)


# =============================================================================
# Q13 – Markov Chain Simulation  (10 marks)
# =============================================================================
print("\n" + "="*60)
print("Q13: Markov Chain Simulation")
print("="*60)
theory_q13 = """
MARKOV PROCESS: A stochastic process where the next state depends ONLY on
the current state, not on the history (memoryless property).

MARKOV PROPERTY: P(X_{n+1} | X_n, X_{n-1}, ..., X_0) = P(X_{n+1} | X_n)
"""
print(theory_q13)

# State: weather (Sunny, Cloudy, Rainy)
states       = ['Sunny', 'Cloudy', 'Rainy']
trans_matrix = np.array([
    [0.7, 0.2, 0.1],   # from Sunny
    [0.3, 0.4, 0.3],   # from Cloudy
    [0.2, 0.3, 0.5],   # from Rainy
])

def simulate_markov(states, trans_matrix, start_state, n_steps=10, seed=0):
    np.random.seed(seed)
    current = states.index(start_state)
    chain = [states[current]]
    for _ in range(n_steps):
        current = np.random.choice(len(states), p=trans_matrix[current])
        chain.append(states[current])
    return chain

chain = simulate_markov(states, trans_matrix, start_state='Sunny', n_steps=15)
print("Simulated Markov Chain (Weather):")
print(" → ".join(chain))


# =============================================================================
# Q14 – End-to-End ML Pipeline  (20 marks)
# =============================================================================
print("\n" + "="*60)
print("Q14: Complete End-to-End ML Pipeline")
print("="*60)

from sklearn.ensemble import RandomForestClassifier

# (i) Reading data
df_pipe = pd.read_csv(DATASET_PATH)
df_pipe.drop(columns=['Id'], inplace=True)
print("Step 1 – Data loaded:", df_pipe.shape)

# (ii) Handling missing values
df_pipe.fillna(df_pipe.median(numeric_only=True), inplace=True)
print("Step 2 – Missing values handled:", df_pipe.isnull().sum().sum())

# (iii) Outlier removal (IQR on PetalLengthCm)
for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    q1, q3 = df_pipe[col].quantile(0.25), df_pipe[col].quantile(0.75)
    iqr = q3 - q1
    df_pipe = df_pipe[(df_pipe[col] >= q1 - 1.5*iqr) & (df_pipe[col] <= q3 + 1.5*iqr)]
print("Step 3 – Outliers removed:", df_pipe.shape)

# (iv) Skewness removal
for col in ['PetalLengthCm', 'PetalWidthCm']:
    df_pipe[col] = np.log1p(df_pipe[col])
print("Step 4 – Skewness removed (log transform applied)")

# (v) Categorical encoding
df_pipe['Species'] = LabelEncoder().fit_transform(df_pipe['Species'])
print("Step 5 – Species encoded")

# (vi) Feature engineering – ratio feature
df_pipe['Sepal_Ratio'] = df_pipe['SepalLengthCm'] / df_pipe['SepalWidthCm']
print("Step 6 – Feature engineering: Sepal_Ratio added")

# (vii) Standard Scaler
X_pipe = df_pipe.drop('Species', axis=1)
y_pipe = df_pipe['Species']
X_tr, X_te, y_tr, y_te = train_test_split(X_pipe, y_pipe, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)
print("Step 7 – Standard Scaler applied")

# (viii) Already split above
print("Step 8 – Train/Test split: 80/20")

# (ix) Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)
print("Step 9 – RandomForest model trained")

# (x) Evaluate
from sklearn.metrics import accuracy_score, classification_report
y_pred_pipe = clf.predict(X_te)
print("Step 10 – Model Evaluation:")
print(f"  Accuracy: {accuracy_score(y_te, y_pred_pipe)*100:.2f}%")
print(classification_report(y_te, y_pred_pipe, target_names=['setosa','versicolor','virginica']))


# =============================================================================
# Q15 – KNN Classification  (marks as per question)
# =============================================================================
print("\n" + "="*60)
print("Q15: KNN Classification Model")
print("="*60)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# (i) Load and preprocess
df_knn = pd.read_csv(DATASET_PATH)
df_knn.drop(columns=['Id'], inplace=True)
df_knn['Species'] = LabelEncoder().fit_transform(df_knn['Species'])

X_knn = df_knn.drop('Species', axis=1)
y_knn = df_knn['Species']

# (ii) Train-test split
X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

sc_k = StandardScaler()
X_tr_k = sc_k.fit_transform(X_tr_k)
X_te_k = sc_k.transform(X_te_k)

# (iii) Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr_k, y_tr_k)
y_pred_k = knn.predict(X_te_k)

# (iv) Evaluation
print(f"Accuracy  : {accuracy_score(y_te_k, y_pred_k)*100:.2f}%")
print(f"Precision : {precision_score(y_te_k, y_pred_k, average='macro'):.4f}")
print(f"Recall    : {recall_score(y_te_k, y_pred_k, average='macro'):.4f}")
print(f"F1-Score  : {f1_score(y_te_k, y_pred_k, average='macro'):.4f}")

cm = confusion_matrix(y_te_k, y_pred_k)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['setosa', 'versicolor', 'virginica'])
disp.plot(cmap='Blues')
plt.title("Q15: KNN – Confusion Matrix")
plt.tight_layout()
plt.savefig("q15_knn_cm.png", dpi=150)
plt.show()


# =============================================================================
# Q15b – Supervised vs Unsupervised ML  (Theory)
# =============================================================================
print("\n" + "="*60)
print("Q15b: Supervised vs Unsupervised ML (Theory)")
print("="*60)
theory_sup = """
SUPERVISED MACHINE LEARNING:
  - Training data has LABELLED examples (input → output pairs).
  - Goal: Learn a mapping f(X) → y.
  - Algorithms: Linear Regression, Decision Trees, KNN, SVM, Neural Networks.
  - Example: Classifying Iris species from measurements (labels provided).

UNSUPERVISED MACHINE LEARNING:
  - Training data has NO labels.
  - Goal: Find hidden patterns or structure.
  - Algorithms: K-Means, DBSCAN, PCA, Autoencoders.
  - Example: Clustering Iris flowers without knowing the species names.
"""
print(theory_sup)


# =============================================================================
# Q16 – Data Types in Python  (Theory)
# =============================================================================
print("\n" + "="*60)
print("Q16: Data Types in Python")
print("="*60)
theory_dtypes = """
Python Built-in Data Types:

1. int        – Integer numbers            e.g., x = 5
2. float      – Decimal numbers            e.g., x = 3.14
3. complex    – Complex numbers            e.g., x = 2 + 3j
4. str        – String (text)              e.g., x = "Iris"
5. bool       – Boolean (True/False)       e.g., x = True
6. list       – Ordered, mutable           e.g., x = [1, 2, 3]
7. tuple      – Ordered, immutable         e.g., x = (1, 2, 3)
8. set        – Unordered, unique items    e.g., x = {1, 2, 3}
9. dict       – Key-value pairs            e.g., x = {'a': 1}
10. NoneType  – Null/missing value         e.g., x = None

In Pandas (for ML):
  int64, float64  → numeric
  object          → string/categorical
  bool            → boolean
  datetime64      → dates and times

Example from Iris dataset:
  SepalLengthCm → float64
  Species       → object (categorical)
"""
print(theory_dtypes)


# =============================================================================
# Q17 – Assumptions of Linear Regression  (Theory)
# =============================================================================
print("\n" + "="*60)
print("Q17: Assumptions of Linear Regression")
print("="*60)
theory_lr = """
LINEAR REGRESSION ASSUMPTIONS:

1. LINEARITY
   - There is a linear relationship between X and y.
   - Violation: scatter plot shows a curve → use polynomial regression.

2. INDEPENDENCE OF ERRORS
   - Residuals are independent of each other.
   - Violation: autocorrelation in time-series data.

3. HOMOSCEDASTICITY
   - Variance of residuals is constant across all X values.
   - Violation: funnel-shaped residual plot → apply log transform.

4. NORMALITY OF RESIDUALS
   - Residuals should be normally distributed.
   - Check: Q-Q plot or Shapiro-Wilk test.

5. NO MULTICOLLINEARITY
   - Independent variables should not be highly correlated with each other.
   - Check: VIF (Variance Inflation Factor) > 10 signals multicollinearity.

6. NO OUTLIERS
   - Extreme outliers heavily influence the regression line.
   - Remove or use robust regression.
"""
print(theory_lr)


# =============================================================================
# Q18 – Evaluation Parameters for Classification  (Theory + Code)
# =============================================================================
print("\n" + "="*60)
print("Q18: Classification Evaluation Parameters")
print("="*60)
theory_eval = """
Given:
  TP = True Positive   (correctly predicted positive)
  TN = True Negative   (correctly predicted negative)
  FP = False Positive  (predicted positive, actually negative)
  FN = False Negative  (predicted negative, actually positive)

1. ACCURACY
   = (TP + TN) / (TP + TN + FP + FN)
   Limitation: misleading for imbalanced datasets.

2. PRECISION
   = TP / (TP + FP)
   "Of all predicted positives, how many are actually positive?"
   Important when cost of False Positive is high (spam detection).

3. RECALL (Sensitivity)
   = TP / (TP + FN)
   "Of all actual positives, how many did we correctly predict?"
   Important when cost of False Negative is high (cancer detection).

4. F1-SCORE
   = 2 * (Precision * Recall) / (Precision + Recall)
   Harmonic mean of precision and recall.
   Best when you need balance between Precision and Recall.

5. CONFUSION MATRIX
   Rows = Actual class, Columns = Predicted class.
   Off-diagonal elements show misclassifications.
"""
print(theory_eval)

# Demonstrate with KNN predictions from Q15
from sklearn.metrics import classification_report
print("\nDetailed Classification Report (from Q15 KNN):")
print(classification_report(y_te_k, y_pred_k,
                            target_names=['setosa', 'versicolor', 'virginica']))

print("\n=== ALL QUESTIONS COMPLETED ===")
print("Plots saved: q2_visualizations.png, q5_outliers.png, q6_skewness.png,")
print("             q8_heatmap.png, q9_pca.png, q11_regression.png, q15_knn_cm.png")
