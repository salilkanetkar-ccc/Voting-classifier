
import pandas as pd
import nltk
import numpy as np
import os
import string
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

N_COMPONENTS = 80
VAL_SIZE = 0.2

root_path = r'C:\Users\skanetkar\Downloads\Sample Problem\Sample Problem'
mapping_address = os.path.join(root_path, r'Interview_Mapping.csv')
judgements_address_root  = os.path.join(root_path, r'Fixed Judgements\Fixed Judgements')

def read_csv(address):
    df = pd.read_csv(address)
    return df

df_mapping = read_csv(mapping_address)


# Now that we have read the CSV mapping file, we will split the labelled and the unlabelled data filenames. We eventually want to classify the unlabelled files to their cases types.
unlabelled_filenames = []
labelled_filenames = []
labels = []

for index,row in df_mapping.iterrows():
    if row['Area.of.Law'] == 'To be Tested':
        unlabelled_filenames.append(row['Judgements'])
    else:
        labelled_filenames.append(row['Judgements'])
        labels.append(row['Area.of.Law'])


# Now that we have the filenames, lets access the judgements file to get the text of the judgements and store seperately  the labelled and the unlabelled judgements.

unlabelled_text = []
labelled_text = []

for name in unlabelled_filenames:
    full_path = os.path.join(judgements_address_root, name + '.txt')
    with open(full_path, 'r', errors = 'ignore') as f:
        unlabelled_text.append(f.read())

for name in labelled_filenames:
    full_path = os.path.join(judgements_address_root, name + '.txt')
    with open(full_path, 'r', errors = 'ignore') as f:
        labelled_text.append(f.read())


# Now that we have extracted the text, we will do some text cleaning like removing links, lemmatization and puntuation removal.
#clean text_list
def clean(list):
    cleaned_list = []
    for element in list:
        cleaned_list.append(clean_text(element))
    return cleaned_list

def initiate_lemmatization():
    normalizer = WordNetLemmatizer()
    return normalizer

def lemmatization_function(lemmatizer):
    lemmatization_func = lemmatizer.lemmatize
    return lemmatization_func

def remove_links(str):
    str = re.sub(r'http(s)?:\/\/\S*', "", str)
    return ''.join([ch if ch.isalnum() or ch.isspace() else " " for ch in str ])

def clean_text(text):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    
    #remove links
    text = remove_links(text)
    
    #remove stopwords, puctuations and make text lower case
    stop_free = ''.join([ch for ch in text if not ch.isdigit()])
    upper_free = " ".join([word for word in stop_free.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in upper_free if ch not in exclude)
    
    #do lemmatization
    lemmatizer = initiate_lemmatization()
    lemmatizer_func = lemmatization_function(lemmatizer)
    lemmatized_text = " ".join(lemmatizer_func(word) for word in punc_free.split())
    return lemmatized_text

cleaned_labelled_text = clean(labelled_text)
cleaned_unlabelled_text = clean(unlabelled_text)


# Now that we have cleaned the text, its time to split the labelled text into training and validation set. We will do 80-20 split for this
train_text, val_text, train_labels, val_labels = train_test_split(cleaned_labelled_text, labels, test_size=VAL_SIZE)


# Now we fit a TF-IDF vectorizer on train text followed by by truncated SVD for dimensionality reduction. Then we will transform the validation text using these trained objects.
def do_TFIDF(train_data, val_data):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_val = vectorizer.transform(val_data)
    return X_train, X_val, vectorizer

X_train, X_val, tfidf_vec = do_TFIDF(train_text, val_text)


# We will do dimensionality reduction using SVD(better than PCA for NLP tasks)
svd = TruncatedSVD(n_components=N_COMPONENTS)
X_train = svd.fit_transform(X_train)
X_val = svd.transform(X_val)


#  we will do Label Encoding on judgement labels
def do_LE(corpus, train_data, val_data):
    LE = preprocessing.LabelEncoder()
    LE.fit(corpus)
    y_train = LE.transform(train_data)
    y_val = LE.transform(val_data)
    all_classes = LE.classes_
    return y_train, y_val, all_classes

y_train, y_val, all_labels = do_LE(labels, train_labels, val_labels)


# Now that we have vecorized our data, its time to train the classifiers. We will train 3 types of classifiers on the training data- Decision Trees, Multiclass Naive Bayes and KNN, to solve this multilabel problem. We will then use them in an ensemble using soft voting and apply it on the unlabelled text for prediction.
classifier_list = [('DTC',DecisionTreeClassifier(random_state=0)), ('KNN',KNeighborsClassifier(n_neighbors=3)),('GNB',GaussianNB())]

def train_and_evaluate(classifier_list, train_data, train_labels, val_data, val_labels):
    classifier = VotingClassifier(estimators=classifier_list, voting = 'soft')
    classifier.fit(train_data, train_labels)
    y_pred = classifier.predict(val_data)
    score = f1_score(val_labels, y_pred, average='micro')
    return score, classifier

scores, trained_classifier = train_and_evaluate(classifier_list, X_train, train_labels, X_val, val_labels)

# Now that we have trained the classifier, its time for final predictions. We have already cleaned the unlabelled text. We will do  TFIDF vectorization and SVD on it.
X_test = tfidf_vec.transform(cleaned_unlabelled_text)
X_test = svd.transform(X_test)


# We will predict the judgement types on the vectorized unlabelled text using the best classifier and write it to a text file

predictions = trained_classifier.predict(X_test)
results_file_address = r'C:\Users\skanetkar\Desktop\results.txt'

with open(results_file_address, 'w') as f:
    f.write('Judgements' + '\t' + 'Area.of.Law' + '\n' )
    for i, judgements in enumerate(unlabelled_filenames):
        f.write(judgements + '\t' + predictions[i] + '\n')

