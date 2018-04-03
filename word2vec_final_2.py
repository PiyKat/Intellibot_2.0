import gensim as gs
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os




def create_model(csv_path):

    #from word_vectors import TfIdfEmbedding
    from word_vectors import create_gensim_word2vec
    from word_vectors import TfIdfEmbedding
    #import word_vectors.TfIdfEmbedding
    w2v,train_data = create_gensim_word2vec("../data/word2vec_final/"+csv_path)
    print(train_data)

    from sklearn.pipeline import Pipeline
    from sklearn.externals import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split,cross_val_score

    #extree_w2v = Pipeline([("word2vec",Word2VecEmbedding(w2v)),("extree",ExtraTreesClassifier(n_estimators=400))])
    extree_w2v_tfidf = Pipeline([("word2vec_tfidf",TfIdfEmbedding(w2v)),("extree",ExtraTreesClassifier(n_estimators = 400))])

    param_grid = {'C': [1e3, 5e3, 2e3,3e3,4e3,1e4, 5e4, 1e5,5e5,1e2,10,1],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  'kernel' : ['rbf','poly','linear','sigmoid']}

    grid_search = GridSearchCV(SVC(class_weight='balanced', probability=True),
                               param_grid, n_jobs=10, verbose=1)

    svm_w2v_tfidf = Pipeline([("word2vec_tfidf",TfIdfEmbedding(w2v)),("svm",grid_search)])

    param_grid_logistic = { 'C' : [1e3, 5e3,2e3,3e3,4e3,1e4, 5e4, 1e5,5e5,1e2,10,1],
                            'solver':['newton-cg','liblinear','lbfgs']}

    log_grid = GridSearchCV(LogisticRegression(penalty="l2"),param_grid_logistic,n_jobs=10,verbose=1)

    lr_w2v_tfidf = Pipeline([("word2vec_tfidf",TfIdfEmbedding(w2v)),("log_reg",log_grid)])


    X_train, X_test, y_train, y_test = train_test_split(train_data.values[:,1],train_data.values[:,2], test_size=0.15)

    #extree_w2v.fit(X,y)
    print("TRAINING THE EXTREE MODEL : \n")
    extree_w2v_tfidf.fit(X_train,y_train)
    print("\n")

    print("TRAINING THE SVM MODEL : \n")
    svm_w2v_tfidf.fit(X_train,y_train)
    print("\n")

    print("TRAINING THE LOG REG MODEL : \n")
    lr_w2v_tfidf.fit(X_train,y_train)


    print("\n")

    with open("../data/word2vec_final/lr_model.pkl","wb") as f:
        joblib.dump(lr_w2v_tfidf,f)
        print(f)
        print(type(f))



    #### Evaluating the models

    svm_validation = cross_val_score(svm_w2v_tfidf,X_train,y_train,scoring='accuracy',cv=5)
    extree_validation = cross_val_score(extree_w2v_tfidf,X_train,y_train,scoring='accuracy',cv=5)
    lr_validation = cross_val_score(lr_w2v_tfidf,X_train,y_train,scoring="accuracy",cv =5)

    print("SVM CROSS VALIDATION : ")
    print(svm_validation)
    print("MEAN CV SCORE : ")
    print(np.mean(svm_validation))
    print("SVM BEST FIT : ")
    print(svm_w2v_tfidf.named_steps['svm'].best_params_)
    svm_w2v_tfidf_pred = svm_w2v_tfidf.predict(X_test)
    print("ACCURACY FOR SVM ON TEST DATA: ", accuracy_score(y_test, svm_w2v_tfidf_pred))
    #evaluate_models(svm_w2v_tfidf, "SVM")


    print("EXTREE CROSS VALIDATION : ")
    print(extree_validation)
    print("MEAN CV SCORE : ")
    print(np.mean(extree_validation))
    extree_w2v_tfidf_pred = extree_w2v_tfidf.predict(X_test)
    print("ACCURACY FOR EXTREE W2V-TFIDIF ON TEST DATA: ", accuracy_score(y_test, extree_w2v_tfidf_pred))
    #evaluate_models(extree_w2v_tfidf, "EXTREE")

    print("LR CROSS VALIDATION : ")
    print(lr_validation)
    print("MEAN CV SCORE : ")
    print(np.mean(lr_validation))
    lr_w2v_tfidf_pred = lr_w2v_tfidf.predict(X_test)
    print("ACCURACY FOR LR ON TEST DATA: ", accuracy_score(y_test, lr_w2v_tfidf_pred))
    print("LR BEST FIT : ")
    print(lr_w2v_tfidf.named_steps['log_reg'].best_params_)




def create_classify_model(csv_path):


    from word_vectors import create_gensim_word2vec
    from word_vectors import TfIdfEmbedding


    from sklearn.externals import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_score


    # import word_vectors.TfIdfEmbedding
    w2v, train_data = create_gensim_word2vec("../data/word2vec_final/" + csv_path)
    #print(train_data)

    X_train, X_test, y_train, y_test = train_test_split(train_data.values[:, 1], train_data.values[:, 2],
                                                        test_size=0.15)

    word2vec_object = TfIdfEmbedding(w2v)

    word2vec_transform = word2vec_object.transform(X_train)



    #### CREATING EXTREE CLASSIFIER

    # extree = ExtraTreesClassifier(n_estimators = 400)
    #
    # extree.fit(word2vec_transform,y_train)
    #
    # print("EXTREE MODEL TRAINED")
    #
    # with open("../data/word2vec_final/extree_model.pkl","wb") as f:
    #     joblib.dump(extree,f)
    #     #print(z)
    #     #print(type(f))
    #
    #
    # print("ACCURACY OF EXTREE ON OUR DATASET : ", accuracy_score(y_test,extree.predict(word2vec_object.transform(X_test))))



    ###### CREATING SVM CLASSIFIERS

    param_grid = {'C': [1e3, 5e3, 2e3, 3e3, 4e3, 1e4, 5e4, 1e5, 5e5, 1e2, 10, 1],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

    grid_search = GridSearchCV(SVC(class_weight='balanced', probability=True),
                               param_grid, n_jobs=-1, verbose=1)

    svm = grid_search.fit(word2vec_transform,y_train)

    with open("../data/word2vec_final/svm_model.pkl","wb") as s:
        joblib.dump(svm,s)

    print("SVM MODEL TRAINED!!!!")

    print("BEST PARAMETERS OF SVM : ")
    print(svm.best_params_)

    print("ACCURACY OF SVM ON TRAINING DATASET: ",accuracy_score(y_train,svm.predict(word2vec_object.transform(X_train))))
    print("ACCURACY OF SVM ON OUR TESTING DATASET : ",accuracy_score(y_test,svm.predict(word2vec_object.transform(X_test))))


    evaluate_models(svm,"SVM",word2vec_object)



    #### CREATING A LOG REG CLASSIFIER

    param_grid_logistic = {'C': [1e3, 5e3, 2e3, 3e3, 4e3, 1e4, 5e4, 1e5, 5e5, 1e2, 10, 1],
                           'solver': ['newton-cg', 'liblinear', 'lbfgs']}

    log_grid = GridSearchCV(LogisticRegression(penalty="l2"), param_grid_logistic, n_jobs=-1, verbose=1)

    log_reg = log_grid.fit(word2vec_transform,y_train)

    with open("../data/word2vec_final/log_reg_model.pkl","wb") as g:
        joblib.dump(log_reg,g)

    print("LOG REG MODEL TRAINED")

    print("BEST LOGREG PARAMETERS : ")
    print(log_reg.best_params_)

    print("ACCURACY OF LOG REG ON OUR TRAINING DATASET : ",accuracy_score(y_train,svm.predict(word2vec_object.transform(X_train))))
    print("ACCURACY OF LOG REG ON OUR TESTING DATASET : ", accuracy_score(y_test,log_reg.predict(word2vec_object.transform(X_test))))






def evaluate_models(model_file,model_name,word2vec_object):

    ques = ["Can you book a flight for me please?",
            "Just book a meeting with my flight manager please",
            "Send a text message not a mail to book a meeting on a flight",
            "Book a meeting",
            "I would like you to book a flight please",
            "Schedule a flight before a meeting",
            "I would like you to schedule a meeting before booking the flight",
            "Book a flight from delhi to mumbai",
            "Send a text to david",
            # "Send a formal mail to mary telling her to come early in the morning",
            "Let sara know i am in delhi via text message",
            # "Mail the students that the class has been canceled today",
            "Schedule a meeting in a flight",
            # "Send an email not a message to book a meeting in the flight",
            # "Remind everyone about the meeting",
            "Can you schedule a meeting with my friend please",
            "book a meeting with david if done book a flight to mumbai",
            # "I don't want you to book a flight",
            "Would you mind turning off the fan and lights",
            "Can you please send an email to Mr. Mogambo",
            "What is the level of precipitation in the Bahamas",
            "I am really bored",
            "Tell me a joke about Mr Donald Trump"
            # "I want you to shut up"
            ]

    gold_standard = ["FLIGHT", "MEETING", "CONNECT", "MEETING", "FLIGHT", "FLIGHT", "MEETING", "FLIGHT", "CONNECT",
                     "CONNECT", "MEETING", "MEETING", "MEETING",
                     "IOT", "CONNECT", "CLIMATE", "JOKE", "JOKE"]


    def calculate_accuracies(answers,model,questions = ques):


        score = 0

        for i in range(len(answers)):

            if answers[i] == gold_standard[i].lower():
                score = score + 1
            else:
                questions[i] = questions[i] + " * "

        print("ACCURACY FOR " + str(model) + " : " + str(score) + " out of " + str(len(answers)))
        print("QUESTIONS WITH * ARE CLASSIFIED WRONG")

        for q in questions:

            print(q + " :: " +  str(answers[questions.index(q)]) + "  CORRECT ANSWER  :  " + gold_standard[questions.index(q)])



    #calculate_accuracies(model_file.predict(word2vec_object.transform(ques)),model_name)


def classify_and_ner(statement):


    from sklearn.externals import joblib

    import numpy as np
    from ner import call_model

    model_file = open("../data/word2vec_final/svm_model.pkl","rb")

    selected_model = joblib.load(model_file)
    statement_vector = np.asarray(statement)
    x = selected_model.predict(statement_vector)
    print(x)
    #return x


    call_model(statement[0],x[0])



def classify(statement):


    from sklearn.externals import joblib
    from word_vectors import TfIdfEmbedding,return_word2vec

    import numpy as np

    w2v = return_word2vec()

    word2vec = TfIdfEmbedding(w2v)

    model_file = open("../data/word2vec_final/svm_model.pkl", "rb")

    #statement_vector = np.asarray(statement)


    user_w2v = word2vec.convert_query(statement)


    selected_model = joblib.load(model_file)

    x = selected_model.predict(user_w2v)

    return x


def agent_statistics():

    dataframe = pd.read_csv("../data/word2vec_final/commands_2.csv")
    dataframe.drop_duplicates(subset=['text'],inplace=True)

    agent_dict = {}

    for column_value in dataframe['agent'].unique():

        agent_dict[column_value] = 0

    for _,row in dataframe.iterrows():

        agent_dict[row['agent']] += 1

    print(agent_dict)



#agent_statistics()



    # def classify_and_ner(statement):
#
#     #from sklearn.feature_extraction.text import TfidfVectorizer
#     os.chdir("..")
#
#     from ner import call_model
#     from sklearn.externals import joblib
#
#     import numpy as np
#
#
#     model_file = open("lr_model.pkl","rb")
#     selected_model = joblib.load(model_file)
#
#     #tf_file = open("tfidf.pkl","rb")
#     #tfidf_trans = pkl.load(tf_file)
#
#
#     statement_vector = np.asarray(statement)
#     print(statement_vector)
#     #x_test_counts = count_vect.fit_transform(statement_vector)
#     #x_test_tfidf = tfidf_transformer.fit_transform(statement_vector)
#     #print(x_test_tfidf)
#     x = selected_model.predict(statement_vector)
#     print(x)
#
#     #call_model(statement[0],x[0])

#create_classify_model("commands_3.csv")
print(classify("set a reminder for 2pm of calling surpeet"))
#classify_and_ner(["Just schedule a meeting for me please","Book a meeting"])




# if len(sys.argv) == 2:
#     if os.path.exists(sys.argv[1]):
#         print("File has been found ! " + sys.argv[1])
#         create_model(sys.argv[1])
#     else:
#         print("Invalid file path")


#     for elements in sys.argv[1:]:
#         if re.findall(".csv",elements):
#             print("Found the file")
#             create_gensim_word2vec(elements)
#         else:
#             create_model()
#
# else:
#     create_model()


#print(classify(""))


#classify_and_ner("Let's book a meeting with Piyush")