# Titanic-Survival-Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('https://www.kaggle.com/datasets/brendan45774/test-file?resource=download')

# Exploratory Data Analysis
print("Dataset Shape:", train_data.shape)
print("\nData Overview:")
print(train_data.head())
print("\nData Information:")
print(train_data.info())
print("\nMissing Values:")
print(train_data.isnull().sum())
print("\nSurvival Rate:", train_data['Survived'].mean())

# Visualizations
plt.figure(figsize=(12, 5))

# Survival count
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count')

# Survival by gender
plt.subplot(1, 3, 2)
sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.title('Survival by Gender')

# Survival by passenger class
plt.subplot(1, 3, 3)
sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.title('Survival by Passenger Class')

plt.tight_layout()
plt.show()

# Additional visualizations
plt.figure(figsize=(12, 5))

# Age distribution
plt.subplot(1, 2, 1)
sns.histplot(train_data['Age'].dropna(), kde=True)
plt.title('Age Distribution')

# Fare distribution
plt.subplot(1, 2, 2)
sns.histplot(train_data['Fare'], kde=True)
plt.title('Fare Distribution')

plt.tight_layout()
plt.show()

# Feature Engineering
# Create a new feature: FamilySize
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# Extract titles from names
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print("\nUnique Titles:", train_data['Title'].unique())

# Simplify titles
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Other",
    "Rev": "Other",
    "Col": "Other",
    "Major": "Other",
    "Mlle": "Miss",
    "Countess": "Other",
    "Ms": "Miss",
    "Lady": "Other",
    "Jonkheer": "Other",
    "Don": "Other",
    "Dona": "Other",
    "Mme": "Mrs",
    "Capt": "Other",
    "Sir": "Other"
}
train_data['Title'] = train_data['Title'].map(title_mapping)

# Create feature for cabin information
train_data['HasCabin'] = (~train_data['Cabin'].isnull()).astype(int)

# Extract Embarked, Pclass, Sex as categorical features
categorical_features = ['Embarked', 'Pclass', 'Sex', 'Title']

# Define numerical features requiring imputation
numerical_features = ['Age', 'Fare']

# Define categorical features to be one-hot encoded
categorical_features_to_encode = ['Embarked', 'Sex', 'Pclass', 'Title']

# Create binary features
binary_features = ['HasCabin', 'IsAlone']

# Define features to use
selected_features = numerical_features + categorical_features + binary_features

# Separate the target variable
X = train_data[selected_features]
y = train_data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features_to_encode),
    ],
    remainder='passthrough'  # This includes the binary features without transformation
)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
if hasattr(model['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    ohe_features = list(model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features_to_encode))
    all_features = numerical_features + ohe_features + binary_features
    
    # Get feature importances
    importances = model['classifier'].feature_importances_
    
    # Display only if lengths match
    if len(all_features) == len(importances):
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

# Grid search for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Final model with best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nFinal Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Function to predict survival for new passengers
def predict_survival(passenger_data):
    """
    Predicts survival for new passenger data
    
    Parameters:
    passenger_data (dict): Dictionary with passenger information
    
    Returns:
    int: Predicted survival (0 = Not Survived, 1 = Survived)
    """
    # Convert to DataFrame
    new_passenger = pd.DataFrame([passenger_data])
    
    # Add engineered features
    new_passenger['FamilySize'] = new_passenger.get('SibSp', 0) + new_passenger.get('Parch', 0) + 1
    new_passenger['IsAlone'] = (new_passenger['FamilySize'] == 1).astype(int)
    
    # Handle Title
    if 'Name' in new_passenger.columns:
        new_passenger['Title'] = new_passenger['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        new_passenger['Title'] = new_passenger['Title'].map(title_mapping)
    else:
        new_passenger['Title'] = 'Mr'  # Default
    
    # Handle Cabin
    new_passenger['HasCabin'] = (~new_passenger['Cabin'].isnull()).astype(int) if 'Cabin' in new_passenger.columns else 0
    
    # Select only the necessary features
    new_passenger = new_passenger[selected_features]
    
    # Make prediction
    prediction = best_model.predict(new_passenger)
    return prediction[0]

# Example usage
new_passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 29,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 100,
    'Embarked': 'C',
    'Name': 'Test, Mrs. Example',
    'Cabin': 'C123'
}

survival = predict_survival(new_passenger)
print(f"\nPredicted Survival for New Passenger: {'Survived' if survival == 1 else 'Not Survived'}")
