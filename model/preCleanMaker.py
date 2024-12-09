import pandas as pd
import random
import numpy as np

df = pd.read_csv("model\ptsd_survey.csv")

columns = df.columns.tolist()


bin = [0,29,44,59,120]
labels = [0,1,2,3]
df['Age'] = pd.cut(df['Age'], bins=bin, labels=labels, right=True)
print(df.head())

def create_random_row(columns):
    row = {}
    for col in columns:
        if random.random() < 0.3:  # 30% chance of making a value missing
            row[col] = np.nan
        else:
            if col == "ID":
                row[col] = random.randint(2001, 3000)  # Unique IDs for new rows
            elif col == "Age":
                row[col] = random.randint(18, 90)  # Random age range
            elif col == "Sex":
                row[col] = random.choice(["M", "F"])  # Random gender
            elif col == "PCL-Score":
                row[col] = random.randint(0, 80)  # Valid PCL-Score range
            elif col == "PTSD":
                row[col] = random.choice(
                    ["PTSD", "Panic Disorder", "Specific Phobia", 
                     "Social Phobia", "Generalized Anxiety Disorder", 
                     "Obsessive-Compulsive Disorder", "Adult Separation Disorder"]
                )
            elif col == "PTSD_Percentile":
                row[col] = random.uniform(0, 1)  # Percentile range
    return row


new_rows = [create_random_row(columns) for _ in range(66)]
new_rows_df = pd.DataFrame(new_rows)


def random_perc_by_category(row):
    pcl_score = row["PCL-Score"]
    if pcl_score <= 31:  # None/Resilience
        return random.uniform(0.001, 0.4999)
    elif 32 <= pcl_score <= 47:  # Mild
        return random.uniform(0.5, 0.666)
    elif 48 <= pcl_score <= 63:  # Moderate
        return random.uniform(0.6777, 0.8333)
    elif pcl_score >= 64:  # Serious
        return random.uniform(0.8444, 1)
    else:
        return random.uniform(0, 1)  # Default fallback
    

def random_age_by_category(row):
    category = row["Age"]
    if category == 0:
        return random.randint(18,29)
    if category == 1:
        return random.randint(30,44)
    if category == 2:
        return random.randint(45,59)
    if category == 3:
        return random.randint(60,90)


def random_score_by_category(row):
    category = row["PCL-Score"]
    if category == "None/Resilience":
        return random.randint(0,31)
    if category == "Mild":
        return random.randint(32,47)
    if category == "Moderate":
        return random.randint(48,63)
    if category == "Serious":
        return random.randint(64,80)


def random_condition_by_category(row):
    category = row["PCL-Score"]
    if category in ["Mild", "Moderate", "Serious"]:
        return "PTSD"
    else:
        disorders = [
            "Panic Disorder",
            "Specific Phobia",
            "Social Phobia",
            "Generalized Anxiety Disorder",
            "Obsessive-Compulsive Disorder",
            "Adult Separation Disorder"
        ]
        return random.choice(disorders)
    

df['Age'] = df.apply(random_age_by_category, axis=1)
df['PTSD'] = df.apply(random_condition_by_category, axis=1)
df['PCL-Score'] = df.apply(random_score_by_category, axis=1)
df['PTSD_Percentile'] = df.apply(random_perc_by_category, axis=1)
df = pd.concat([df, new_rows_df], ignore_index=True)



df['ID'] = df['ID'].fillna(0).astype(int)  # Handle NaN values before casting
df['Age'] = df['Age'].fillna(0).astype(int)
df['PCL-Score'] = df['PCL-Score'].fillna(0).astype(int)


df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
df['ID'] = range(1, len(df) + 1)

df['PTSD_Percentile'] = df['PTSD_Percentile'].round(2)

df.to_csv('model/harvard_anxiety_disorders_survey.csv', index=False)

print(df.head())
