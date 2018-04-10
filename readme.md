# Intellibot 2.0

This is the second version of the Intellibot program that leverages the machine learning techniques. This system is much more flexible that the previous version since we are not relying on NLP grammar techniques. 


# Requirements

The system is compatible with Python 3.0+. Kindly refer to the requirements.txt file. The most important points for the format in which data is required is mentioned in the next subheading.


## Accepted Data Format

There are essentially two files required by the program. A <B>commands.csv</B> file that is required by the Novelty Detection module, Text Classification Module as well as the NER module(details about these modules in the next section) and an <B>entities.csv</B> file that is required by the NER module for each service. 

The commands.csv file should have three headers - ID,TEXT,AGENT.

	1. TEXT contains the service commands.
	2. ID is the unique ID of each service command.
	3. AGENT is the service tag which indicates what the user intended.

The entities.csv file should have four headers - ID,ENTITY,START,END.

	1. ID is the ID of the service call in commmands.csv. In this case, it is NOT UNIQUE because a single sentence can have multiple entities.
	2. ENTITY is the entity tag.
	3. START & END specify the range within which the particular entity is present


# PROJECT DESCRIPTION

The NLP Pipeline consists of 3 sequential steps - Novelty detection, Text Classification and Named Entity Recognition using MITIE library.

	1. The Novelty Detection module is used to find out outliers, which in our case are sentences which are not relevant to our services/not a service call. If 		relevant, we return +1 and call the text classification. The novelty detection is created using oneClassSVM algorithm.
	
	2. The Text Classification classifies the text from the user into pre-defined services. The Text Classification uses a standard SVM, the parameters of which have 		been optimized using GridSearchCV, with a custom vectorizer that uses sentence word vectors, calculated by averaging the word vectord of each word. The word 		vectors of the words are calculated used the gensim module(refer to the word_vectors.py file for more details).

	3. The Named Entity Recognition module uses the MITIE system, the bare essential requirements of which are present in the lib folder. Each service requires it's 		own NER model.
