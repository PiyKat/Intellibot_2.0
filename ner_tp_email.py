from time import time
import sys,os
from nltk import word_tokenize
import pandas as pd
import sys

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent+"/../lib/ext_lib/mitielib")

from mitie import *


def feature_extractor():

    #start_feature = time()
    trainer = ner_trainer("../data/ext_lib/total_word_feature_extractor.dat")
    #print("FEATURE IMPORT TIME : ",time()-start_feature)

    return trainer


def ner_trainer_function(class_sheet_path,entity_sheet_path):

    class_df = pd.read_csv(class_sheet_path,names = ["id","text","label"],sep=",",header=0)
    entity_df = pd.read_csv(entity_sheet_path, names=["id", "entity", "start", "end"], sep=",", header=0)

    return class_df,entity_df


def create_model(class_path,entity_path):

    #print("Started the model training for the " + service.upper() + " service.")

    c_df, e_df = ner_trainer_function(class_path,entity_path)
    #new_trainer = feature_extractor()


    information = ["d"]

    for info in information :

        print("Executing for " + info)

        new_trainer = feature_extractor()

        for ids in c_df['id']:

            #print("IDS : " + str(ids))
            sub_df = e_df[e_df['id'] == ids]
            #print(sub_df)

            if len(sub_df) != 0:

                articles = ['a','an','the']

                sample_text = c_df[c_df['id'] == ids]['text'].values[0] +"."
                sample = ner_trainer_function([x.lower() for x in word_tokenize(sample_text) if x not in articles])


                #sample = ner_training_instance([x for x in word_tokenize(c_df[c_df['id'] == ids]['text'].values[0] +".") if x not in articles])

                for i,row in sub_df.iterrows():
                    #print(i)
                    try :
                        #print(row['entity'][-1])
                        if row['entity'][0] == info:
                            #print(row['entity'])
                            sample.add_entity(range(int(row['start']),int(row['end'])) ,row['entity'])

                            new_trainer.add(sample)
                    except:
                        print("Problem in ID : ",str(ids))

        new_trainer.num_threads = 16
        ner_climate = new_trainer.train()
        ner_climate.save_to_disk("../data/ner_tp_email/"+info + "_ner_model.dat")



def call_model(user_prompt):

    ner = named_entity_extractor("../data/ner_tp_email/d_ner_model.dat")

    #user_dict = dictionary_creator(classification,ner)

    tokens = word_tokenize(user_prompt)
    entities1 = ner.extract_entities(tokens)
    print(entities1)

    #user_dict = dictionary_creator(classification,ner)

    for e in entities1:
        a = e[0]
        b = e[1]
        entity = " ".join(tokens[i] for i in a)

        print("ENTITY : " + str(b) + " VALUE : " + entity)



##### A function created to calculate the statistics for our entities

def label_statistics():

    e_df = pd.read_csv("../data/entities.csv",names=["id","entity","start","end"],sep=",",header = 0)
    c_df = pd.read_csv("../data/text.csv",names=["id","text","agent"],sep=",",header=0)

    information = ['d','t','l']


    stat_dict = {'dp':{'count':0,'multi':0,'single':0},
                 'dn':{'count':0,'multi':0,'single':0},
                 'tp':{'count':0,'multi':0,'single':0},
                 'tn':{'count':0,'multi':0,'single':0},
                 'lp':{'count':0,'multi':0,'single':0},
                 'ln':{'count':0,'multi':0,'single':0}}

    for info in information:

        curr_id = 0

        for i,row in e_df.iterrows():

            if row['entity'][0] == info:

                stat_dict[row['entity']]['count'] += 1

                c_row = c_df[c_df['id'] == row['id']]

                if str(c_row['text']).count(".") >= 2:

                    stat_dict[row['entity']]['multi'] += 1
                    curr_id = row['id']

                else:

                    stat_dict[row['entity']]['single'] += 1
                    curr_id = row['id']


    return stat_dict




create_model("../data/200_text.csv","../data/200_entities.csv")

#dictionary = label_statistics()
#print(dictionary['dn'])
#print(dictionary['dp'])

call_model("I can't come on the 24th. Let's shift it to 28th instead")