import pandas as pd
from sklearn.pipeline import Pipeline
#from word2vec_final import classify_and_ner
#from word_vectors import TfIdfEmbedding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.externals import joblib
import os

from nltk.corpus import stopwords

def train_classifier(csv):

    # preparing training data
    #csv_path = os.chdir("/data/ND/")
    #print(csv_path)

    df = pd.read_csv("../data/novelty/" + csv)

    df2 = pd.read_csv("../data/" + "200_text.csv")

    df = df.append(df2)

    #print(len(df))

    #print(len(df))
    X_train = df.text.values

    # removing stop words
    #custom = ['a','an','the']
    exclude_words = stopwords.words("english")

    try :

        pipe = Pipeline([("vectorizer", CountVectorizer(max_df=0.5, max_features=None,
                             ngram_range=(1,2),stop_words=exclude_words)),("oneClass",
                                svm.OneClassSVM(nu=0.1, kernel='linear',gamma=0.5, verbose=True))])  ###gamma=float(1/len(df)),
        pipe.fit(X_train)
        print("The ND classifier has been trained")

    except:

        print("There is a problem here")
    """ 
    Saving pipeline
    This will replace the previously saved pipeline(pickle)
    """

    with open("../data/novelty/NDpipeline.pkl", "wb") as f:
        joblib.dump(pipe, f)
    
    #return pipe

def load_pipeline():

    pipe_file = open("../data/novelty/NDpipeline.pkl", "rb")
    pipe = joblib.load(pipe_file)
    return pipe

def label_query(user_query):

    pipeline = load_pipeline()
    result = pipeline.predict(user_query)
    return result

def decider(user_query):

    novelty = label_query(user_query)

    if novelty == 1 :
        return 1

    else:
        return 0
        #print("OUTLIER")


##### Possible Solutions - 1. Remove ND 2. Add GridCV(Which is not correct I believe)


#train_classifier("commands_3.csv")


#st = ["Who am I"]
pipe = load_pipeline()
print(pipe.predict(["tomorrow 5 p.m. at CCD"]))
