
# Importing the libraries
import numpy as np
import pandas as pd
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Download NLTK resources
nltk.download('stopwords')

# Importing and processing the dataset
def read_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


# Load JSON data
json_data = read_json_lines('/Sarcasm_Headlines_Dataset.json')  # Adjust the path as needed

# Convert JSON data to DataFrame
dataset = pd.json_normalize(json_data)

# Preview the dataset
print(dataset.head())


print("Columns in the dataset:", dataset.columns)

# Ensure correct column names
text_column = 'headline'  # Adjust if necessary
label_column = 'is_sarcastic'  # Adjust if necessary

# Check if the columns exist
if text_column not in dataset.columns or label_column not in dataset.columns:
    raise ValueError(f"Columns '{text_column}' or '{label_column}' are not found in the dataset")


# Cleaning the texts
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset[text_column][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['is_sarcastic'].values  # Adjust 'label' to the actual name of the target column

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Training and evaluating the Naive Bayes model
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Confusion Matrix:\n", cm_nb)
print("Naive Bayes Accuracy:", accuracy_nb)

# Visualizing results (optional, for the first 10 predictions)
print(np.concatenate((y_pred_nb[:10].reshape(len(y_pred_nb[:10]),1), y_test[:10].reshape(len(y_test[:10]),1)),1))



# Training and evaluating the SVM model
classifier_svm = SVC(kernel='linear', random_state=0)
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Confusion Matrix:\n", cm_svm)
print("SVM Accuracy:", accuracy_svm)

# Visualizing results (optional, for the first 10 predictions)
print(np.concatenate((y_pred_svm[:10].reshape(len(y_pred_svm[:10]),1), y_test[:10].reshape(len(y_test[:10]),1)),1))


# Training and evaluating the Random Forest model
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Visualizing results (optional, for the first 10 predictions)
print(np.concatenate((y_pred_rf[:10].reshape(len(y_pred_rf[:10]),1), y_test[:10].reshape(len(y_test[:10]),1)),1))

