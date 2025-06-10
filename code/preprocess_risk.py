
import pandas as pd
import os

def preprocess_initial(input_path, output_path):
    df = pd.read_csv(input_path)

    # Assign student_id to every group of 5 rows
    df['student_id'] = 'Student ' + (df.index // 5 + 1).astype(str)

    # Track whether each drawing shows any negative emotion
    df['has_negative_emotion'] = df[['Angry', 'Fear', 'Sad']].sum(axis=1) > 0

    # Optional: convert boolean to int for CSV (True/False → 1/0)
    df['has_negative_emotion'] = df['has_negative_emotion'].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete: {output_path}")

def label_risk(group):
    """
    Applies the 60% rule for negative emotions:
    If more than 3 out of 5 drawings show any negative emotion → at risk.
    """
    negative_ratio = group['has_negative_emotion'].sum() / len(group)
    return 1 if negative_ratio > 0.6 else 0

def preprocess(input_path, output_path):
    """
    Aggregates emotion features per student and assigns at-risk label.
    """
    df = pd.read_csv(input_path)

    if 'student_id' not in df.columns or 'has_negative_emotion' not in df.columns:
        raise ValueError("Missing columns'.")

    grouped = df.groupby('student_id')

    records = []
    for student_id, group in grouped:
        emotion_avg = group[['Angry', 'Fear', 'Happy', 'Sad']].mean()
        label_encoded = label_risk(group)
        record = {'student_id': student_id, **emotion_avg.to_dict(), 'label_encoded': label_encoded}
        records.append(record)

    processed_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Aggregated student data saved to: {output_path}")

# Step 1: Add student_id and flag emotional negativity
preprocess_initial('files/testing_onehot.csv', 'files/testing_for_risk.csv')

# Step 2: Aggregate per student and assign risk label
preprocess('files/testing_for_risk.csv', 'files/aggregated_risk_data.csv')