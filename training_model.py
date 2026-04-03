import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("C:/recommendation_system/dataset/styles.csv", on_bad_lines='skip')

# Clean columns
df.columns = df.columns.str.strip()

# Required columns
required_cols = ['id','gender','masterCategory','subCategory','articleType','baseColour','season','usage']
df = df[[col for col in required_cols if col in df.columns]]

# Drop missing values
df.dropna(inplace=True)

# Convert to string
for col in df.columns:
    df[col] = df[col].astype(str)

# Create feature column
df['features'] = (
    df['gender'] + " " +
    df['masterCategory'] + " " +
    df['subCategory'] + " " +
    df['articleType'] + " " +
    df['baseColour'] + " " +
    df['season'] + " " +
    df['usage']
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Save
pickle.dump(df, open("C:/recommendation_system/products.pkl", "wb"))
pickle.dump(tfidf, open("C:/recommendation_system/tfidf.pkl", "wb"))
pickle.dump(tfidf_matrix, open("C:/recommendation_system/tfidf_matrix.pkl", "wb"))

print("✅ Training complete")