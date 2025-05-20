#The code converts emotion labels (Angry, Happy, Sad, Fear) into integer encoding (Label Encoding) and binary vectors (One-Hot Encoding), saving the processed data as training_encoded.csv, testing_encoded.csv, training_onehot.csv, and testing_onehot.csv for model training and testing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

# Paths to CSV files
data_dir = 'ECS-111-Emotion-Detection/data'
train_csv = os.path.join(data_dir, 'training.csv')
test_csv = os.path.join(data_dir, 'testing.csv')

# Output paths
train_encoded_csv = os.path.join(data_dir, 'training_encoded.csv')
test_encoded_csv = os.path.join(data_dir, 'testing_encoded.csv')
train_onehot_csv = os.path.join(data_dir, 'training_onehot.csv')
test_onehot_csv = os.path.join(data_dir, 'testing_onehot.csv')

# Load datasets
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels for Label Encoding
train_data['label_encoded'] = label_encoder.fit_transform(train_data['label'])
test_data['label_encoded'] = label_encoder.transform(test_data['label'])

# Save Label Encoded Data
train_data.to_csv(train_encoded_csv, index=False)
test_data.to_csv(test_encoded_csv, index=False)

# Initialize OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# One-hot encoding
train_onehot = onehot_encoder.fit_transform(train_data[['label_encoded']])
test_onehot = onehot_encoder.transform(test_data[['label_encoded']])

# Convert to DataFrame and concatenate with original data
train_onehot_df = pd.DataFrame(train_onehot, columns=label_encoder.classes_)
test_onehot_df = pd.DataFrame(test_onehot, columns=label_encoder.classes_)

train_data = pd.concat([train_data, train_onehot_df], axis=1)
test_data = pd.concat([test_data, test_onehot_df], axis=1)

# Save OneHot Encoded Data
train_data.to_csv(train_onehot_csv, index=False)
test_data.to_csv(test_onehot_csv, index=False)

print('Data encoding completed. Files saved as training_encoded.csv, testing_encoded.csv, training_onehot.csv, testing_onehot.csv')
