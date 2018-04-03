import gensim as gs


class tryout():

    def __init__(self,csv_path):

        self.csv_path = csv_path


    def list_of_list(self):

        import csv
        from nltk import word_tokenize

        list_of_list = []

        with open(self.csv_path,"rt") as f:

            reader = csv.reader(f)

            for row in reader:

                if row[1] != "text":

                    list_of_list.append(word_tokenize(row[1]))

        return list_of_list



x = tryout(csv_path="../data/word2vec_final/commands.csv")
print(x.list_of_list())
