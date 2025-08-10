import PreProcessing
import FeatureExtraction
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
svm_pipeline = Pipeline([
        ('svmCV',FeatureExtraction.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(PreProcessing.train_news['Statement'],PreProcessing.train_news['Label'])
predicted_svm = svm_pipeline.predict(PreProcessing.test_news['Statement'])
np.mean(predicted_svm == PreProcessing.test_news['Label'])
def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(PreProcessing.train_news):
        train_text = PreProcessing.train_news.iloc[train_ind]['Statement'] 
        train_y = PreProcessing.train_news.iloc[train_ind]['Label']
    
        test_text = PreProcessing.train_news.iloc[test_ind]['Statement']
        test_y = PreProcessing.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(PreProcessing.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
build_confusion_matrix(svm_pipeline)
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureExtraction.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(PreProcessing.train_news['Statement'],PreProcessing.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(PreProcessing.test_news['Statement'])
np.mean(predicted_svm_ngram == PreProcessing.test_news['Label'])
build_confusion_matrix(svm_pipeline_ngram)
print('SVM')
print(classification_report(PreProcessing.test_news['Label'], predicted_svm_ngram))
import pandas as pd
import numpy as np
truenews = pd.read_csv('true.csv')
fakenews = pd.read_csv('fake.csv')
truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac = 1) #Shuffle 100%
from nltk.corpus import stopwords
import string

def process_text(s):

   
    nopunc = [char for char in s if char not in string.punctuation]

    
    nopunc = ''.join(nopunc)
    
    
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string

news['Clean Text'] = news['Article'].apply(process_text)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])
print('proposed model')
print(len(bow_transformer.vocabulary_)) 
news_bow = bow_transformer.transform(news['Clean Text'])
print('Shape of Sparse Matrix: ', news_bow.shape)
print('Amount of Non-Zero occurences: ', news_bow.nnz)
sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)
print(news_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])
predictions = fakenews_detect_model.predict(news_tfidf)
print(predictions)
from sklearn.metrics import classification_report
print (classification_report(news['True/Fake'], predictions))
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

print(len(news_train), len(news_test), len(news_train) + len(news_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()),  
])
pipeline.fit(news_train,text_train)
predictions = pipeline.predict(news_test)
print(classification_report(predictions,text_test))
##############################################################
import pickle
var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))

def detecting_fake_news(var):    

    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("The given statement is ",prediction[0]),
        print("The truth probability score is ",prob[0][1]))


if __name__ == '__main__':
    detecting_fake_news(var)

