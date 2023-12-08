# TERMINAL COMMAND: maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir
# *right now I am trying it with ./maxent_tagger.sh test.word_pos test_file rare_thres feat_thres output_dir

import sys
import os 

def read_from_commandline():
    try:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        rare_thres = sys.argv[3]
        feat_thres = sys.argv[4]
        output_dir = sys.argv[5]

    except IndexError:
        print("Not enough command-line arguments. Correct usage: maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir")
        exit()

    return train_file, test_file, rare_thres, feat_thres, output_dir


class POS_Tagger(object):

    def __init__(self, train_file, test_file, rare_thres, feat_thres, output_dir):
        self.train_file = train_file 
        self.test_file = test_file 
        self.rare_thres = rare_thres 
        self.feat_thres = feat_thres
        self.output_dir = output_dir 

    

    def create_train_voc(self):

        list_of_word_tag_tuples = []
        self.word_freq_dict = {}
        self.indexed_word_dict = {} #dict that maps a unique index of a word to a list of its neighboring words
        # { i : w_i-2, w_i-1, w_i, w_i+1, w_i+2}
        self.indexed_tag_list = {}
            
        ######################## Create train_voc with words and frequencies ###################################

        #### Create a list of tuples with word, tag
        with open(self.train_file, 'r') as file:
            for line in file:
                words_tags = line.rstrip().split()

                for element in words_tags:
                    try:
                        word, tag = element.split('/')
                    # This is so words with \/ are handled correctly. 
                    except ValueError:
                        word1, word2, tag = element.split('/')
                        word = word1 + word2

                    # Replace , # : in text with comma hash colon
                    if word == ',':
                        word = 'comma'
                        tag = 'comma'
                    if word == ':':
                        word = 'colon'
                        tag = 'colon'
                    if word == '#':
                        word = 'hash'
                        tag = 'hash'
                    
                    word_tag_tuple = (word, tag)
                    list_of_word_tag_tuples.append(word_tag_tuple)

        #### Create a dictionary with word:freq



        for i in range(len(list_of_word_tag_tuples)):
            word = list_of_word_tag_tuples[i][0]
            tag = list_of_word_tag_tuples[i][1]

            if word in self.word_freq_dict:
                self.word_freq_dict[word] += 1
            else:
                self.word_freq_dict[word] = 1

            try:
                tag_minus_1 = list_of_word_tag_tuples[i-1][1]
                word_minus_1 = list_of_word_tag_tuples[i-1][0]
            except IndexError:
                tag_minus_1 = None 
                word_minus_1 = None

            try:
                tag_minus_2 = list_of_word_tag_tuples[i-2][1]
                word_minus_2 = list_of_word_tag_tuples[i-2][0]
            except IndexError:
                tag_minus_2 = None 
                word_minus_2 = None

            try:
                word_plus_1 = list_of_word_tag_tuples[i+1][0]
            except IndexError:
                word_plus_1 = None 

            try:
                word_plus_2 = list_of_word_tag_tuples[i+2][0]
            except IndexError:
                word_plus_2 = None

            tags_list = [tag_minus_2, tag_minus_1, tag]
            words_list = [word_minus_2, word_minus_1, word, word_plus_1, word_plus_2]

            self.indexed_tag_list[i] = tags_list
            self.indexed_word_dict[i] = words_list

            #TODO: make some kind of dictionary that maps EACH TOKEN (not each type!) to these features
            
            
            

        #### Sort the dictionary in a list of tuples with word, freq
        sorted_word_freq_dict = sorted(self.word_freq_dict.items(), key=lambda x: x[1], reverse=True)

        output_file_path = self.output_dir + "/train_voc"

        #### Print sorted word, freq to train_voc file
        # *We need to write this file in the output_dir folder. Right now it is just creating it in the current folder. <- DONE :)
        with open(output_file_path, 'w') as file:
            for element, count in sorted_word_freq_dict:
                file.write(f"{element}\t{count}\n")


    def create_vectors(self):

        self.words_rarity = {}
        self.word_contains_number = {}
        self.word_contains_uppercase = {}
        self.word_contains_hyphen = {}


    def isRare(word):
        if self.word_freq_dict[word] < self.rare_thres:
            return True

        else:
            return False 

    def containsNumber(word):
        return any(i.isdigit() for i in word) 

    def containsUpper(word):
        return any(i.isupper() for i in word)

    def containsHyphen(word):
        return any((i == "-") for i in word)


        
     

        


def main():
    train_file, test_file, rare_thres, feat_thres, output_dir = read_from_commandline()
    
    tagger = POS_Tagger(train_file, test_file, rare_thres, feat_thres, output_dir)

    tagger.create_train_voc()
    tagger.create_vectors()



if __name__ == "__main__":
    main()

####################### Debugging, delete later ##########################

#print(sorted_word_freq_dict)
#print(word_freq_dict)
#print(list_of_word_tag_tuples)

        #print(words_tags)
        #print(line)


