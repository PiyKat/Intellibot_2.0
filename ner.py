import sys,os
from nltk import word_tokenize
#from nltk.corpus import stopwords
import pandas as pd
import sys


parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent+"/../lib/ext_lib/mitielib")

from mitie import *


def feature_extractor():

    trainer = ner_trainer("../data/ext_lib/total_word_feature_extractor.dat")

    return trainer

'''   

class ner_trainer(object):

    def __init__(self,class_sheet_path,entity_sheet_path):
        self.class_sheet_path=class_sheet_path
        self.entity_sheet_path = entity_sheet_path

    def parse_csv(self):
        
        class_df = pd.read_csv(self.class_sheet_path,names = ["id","text",'agent'],sep=",",header=0)
        entity_df = pd.read_csv(self.entity_sheet_path, names=["id", "entity", "start", "end"], sep=",", header=0)
        return class_df,entity_df
    

    def train_model(self):
        
 '''

def ner_trainer_function(class_sheet_path,entity_sheet_path):

    class_sheet_path = "../data/ner/" + class_sheet_path
    entity_sheet_path = "../data/ner/" + entity_sheet_path

    class_df = pd.read_csv(class_sheet_path,names = ["id","text",'agent'],sep=",",header=0)
    entity_df = pd.read_csv(entity_sheet_path, names=["id", "entity", "start", "end"], sep=",", header=0)

    ###### Training for all possible models
    #for classes in class_df.agent.unique():
    #    create_model_2(class_df[class_df["agent"]==classes],entity_df,classes)

    ### temporary solution
    create_model_2(class_df[class_df["agent"] == "meeting"],entity_df,"meeting")


def text_preprocess(statement):

    import re

    detect_flag = -1

    remove_words = ['a','an','the']

    modif = ""

    if re.findall("[\d][\s]p\.m\.",statement) != []:

        detect_flag = 1

        modif = re.sub("p\.m\.","pm",statement)

    if re.findall("[\d][\s]a\.m\.",statement) != []:

        detect_flag = 1

        modif = re.sub("a\.m\.","am",statement)

    if detect_flag == -1 :

        return [x.lower() for x in statement.split() if x not in remove_words]

    else:

        return [x.lower() for x in modif.split() if x not in remove_words]



def create_model_2(c_df,e_df,service):

    print("Training for " + service)

    c_df = c_df.sample(frac=1)

    trainer = feature_extractor()

    for ids in c_df['id']:

        #print("IDS : " + str(ids))
        sub_df = e_df[e_df['id'] == ids]
        #print(sub_df)

        if len(sub_df) != 0:

            sample_text = text_preprocess(c_df[c_df['id'] == ids]['text'].values[0])
            print("SAMPLE : " + " ".join(sample_text))

            #text_data = [x.lower() for x in word_tokenize(sample)]

            sample = ner_training_instance(sample_text)

            for i,row in sub_df.iterrows():
                #print(i)

                try :

                    print("ENTITY : " + str(row['entity'] + " VALUE : " + str(sample_text[int(row['start']) : int(row['end'])])))

                    sample.add_entity(range(int(row['start']),int(row['end'])),row['entity'])
                    trainer.add(sample)
                except Exception as e:
                    print("Error in " + str(ids))
                    print(e)


    trainer.num_threads = 16
    ner = trainer.train()
    ner.save_to_disk("../data/ner/" + service + "_model.dat")



def dictionary_creator(classify,ner_model):

    labels = ner_model.get_possible_ner_tags()

    none_list = [None]*len(labels)

    tuples_list = list(zip(labels,none_list))

    create_dict = {"service" : classify,"values": dict(tuples_list) }

    return create_dict



def call_model(user_prompt,classification):

    ner = named_entity_extractor("../data/ner/" + classification + "_model.dat")

    tokens = word_tokenize(user_prompt)
    entities1 = ner.extract_entities(tokens)
    print(entities1)

    user_dict = dictionary_creator(classification,ner)

    counter = 0

    for e in entities1:
        a = e[0]
        b = e[1]
        entity = " ".join(tokens[i] for i in a)

        entity_list = []
        entity_list.append(entity)

        for previous in entities1[:entities1.index(e)]:
            ran = previous[0]
            entity_prev = previous[1]

            if b == entity_prev:
                counter += 1
                entity_list.append(" ".join(tokens[i] for i in ran))
                #print(entity_list)

        if counter >= 1:
            user_dict["values"][b] = entity_list

        else:
            user_dict["values"][b] = entity


    print(user_dict)





def entity_statistic(text_path,entity_path):

    e_df = pd.read_csv(entity_path, names=["id", "entity", "start", "end"], sep=",", header=0)
    c_df = pd.read_csv(text_path, names=["id", "text", "agent"], sep=",", header=0)

    stat_dict = {'name':0,'date':0,'time':0,'loc':0}

    for i,row in e_df.iterrows():

        stat_dict[row["entity"]] += 1


    print(stat_dict)


#print(text_preprocess("Register my appointment with Mr. Sudhir Kalra on 14th may at Karol Bagh at 6 o'clock"))

#entity_statistic("../data/ner/meeting_text_5.csv","../data/ner/meeting_entities_5.csv")

#ner_trainer_function("meeting_text_5.csv","meeting_entities_5.csv")

#"book a meeting on thursday at 5pm with harpreet","book a meeting with Piyush for tomorrow 5pm at sainik vihar",
# "let's schedule a meeting with rahul for 8am tommorow at najafgarh","schedule a meeting with hamza at firangchowk on 26th December at 9am",

# list_of_commands = ["book a meeting with Piyush at my office","book a meeting on thursday at 5 p.m. with harpreet","book a meeting with Piyush for tomorrow 5 p.m. at sainik vihar",
# "lets schedule a meeting with rahul for 8 a.m. tommorow at najafgarh.","schedule a meeting with hamza at firangchowk on 26th December at 9 a.m.",
#                     "lets schedule a meeting with harpeet at my office on 5pm 26th October"
# ]

def testing_ner(sheet_path):

    test_df = pd.read_csv(sheet_path,sep=",",header=0)

    for i,row in test_df.iterrows():

        print("TESTED SENTENCE :"," ".join(text_preprocess(row['text'])))
        call_model(" ".join(text_preprocess(row['text'])),"meeting")
        print("\n")

#testing_ner("../data/ner/testing_statements.csv")


call_model(" ".join(text_preprocess("Book a meeting with veerat for 5 p.m. tomorrow")),"meeting")


#from word2vec_final import classify

# def ner_try(user_array):
#
#     master_query = []
#
#     for user_queries in user_array:
#
#         classification = classify(user_queries)
#
#         user_dict = dictionary_creator(classification, ner)
#
#
#
#
#

    # if classification == "meeting":
    #
    #     user_dict = {"service":classification,
    #                  "values":{"date" : None,
    #                            "time" : None,
    #                            "place" : None,
    #                            "person" : None}
    #                  }
    #     counter = 0
    #
    #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a)
    #
    #         entity_list = []
    #         entity_list.append(entity)
    #
    #         for previous in entities1[:entities1.index(e)]:
    #             ran = previous[0]
    #             entity_prev = previous[1]
    #
    #             if b == entity_prev:
    #                 counter += 1
    #                 entity_list.append(" ".join(tokens[i] for i in ran))
    #                 print(entity_list)
    #
    #         if counter >= 1:
    #             user_dict["values"][b] = entity_list
    #
    #         else:
    #             user_dict["values"][b] = entity



        # for e in entities1:
        #     a = e[0]
        #     b = e[1]
        #     entity = " ".join(tokens[i] for i in a)
        #
        #     for previous in entities1[:entities1.index(e)]:
        #         ran = previous[0]
        #         entity_prev = previous[1]
        #         if b == entity_prev:
        #             counter+=1
        #             entity_list = []
        #             entity_list.append(entity)
        #             entity_list.append(" ".join(tokens[i] for i in ran))
        #             user_dict["values"][b] = entity_list
        #
        #
        #     if counter == 0:
        #
        #         user_dict["values"][b] = entity


    # elif classification == "flight":
    #
    #     user_dict = {"service" : classification,
    #                  "values":{"dest":None,
    #                            "time":None,
    #                            "source":None,
    #                            "class":None,
    #                            "date":None,
    #                            "passengers":None}
    #                  }
    #
    #     counter = 0
    #
    #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a)
    #
    #         entity_list = []
    #         entity_list.append(entity)
    #
    #         for previous in entities1[:entities1.index(e)]:
    #             ran = previous[0]
    #             entity_prev = previous[1]
    #
    #             if b == entity_prev:
    #                 counter += 1
    #                 entity_list.append(" ".join(tokens[i] for i in ran))
    #                 print(entity_list)
    #
    #         if counter >= 1:
    #             user_dict["values"][b] = entity_list
    #
    #         else:
    #             user_dict["values"][b] = entity
    # #     for e in entities1:
    #         a = e[0]
    #         b = e[1]
    #         entity = " ".join(tokens[i] for i in a )
    #         user_dict["values"][b] = entity
    #
    # else:
    #     print("Service currently not available")
