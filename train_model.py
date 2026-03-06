import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier





df = pd.read_csv("customer_data.csv")

# defining features and target

features = df.drop (columns= ["exited"])

target = df["exited"]

# split and train 

X_train , y_train , X_test , y_test = train_test_split (
    features , target ,
    test_size=0.2,
    random_state=42
)
 
# categorizing data types

categorical_cols = features.select_dtypes(include=["object" , "string"]).columns
numerical_cols = features.select_dtypes(include= ["number"]) 

# column transfer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Defining Model

model = Pipeline (steps = [
    ("preprocessor" , preprocessor),
    ("Classifier" , RandomForestClassifier(random_State = 42))
] )


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

