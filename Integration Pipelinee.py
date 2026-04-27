import pandas as pd
import string
import re
from nltk.stem import WordNetLemmatizer

df = pd.read_excel("training_data_merged.xlsx")
lemmatizer = WordNetLemmatizer()


def to_lower(text):
    return text.lower()


def clean_text(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[%$#@]', '', text)
    return text


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_pipeline(text):
    text = to_lower(text)
    text = clean_text(text)
    text = lemmatize_text(text)
    return text

df['clean_text'] = df['Incident_text'].apply(preprocess_pipeline)

df = df.drop_duplicates()


print("Class Distribution:")
print(df["Category"].value_counts())
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_excel("final_cleaned_data.xlsx", index=False)

print("Pipeline finished successfully!")
