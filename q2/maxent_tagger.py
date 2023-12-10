# TERMINAL COMMAND: maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir
# *right now I am trying it with ./maxent_tagger.sh test.word_pos test_file rare_thres feat_thres output_dir

import sys
import os 
from collections import defaultdict

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


#######POS TAGGER CLASS#######

class POS_Tagger(object):

    def __init__(self, train_file, test_file, rare_thres, feat_thres, output_dir):
        self.train_file = train_file 
        self.test_file = test_file 
        self.rare_thres = int(rare_thres) 
        self.feat_thres = int(feat_thres)
        self.output_dir = output_dir 


    def create_train_voc(self):
        list_of_word_tag_tuples = []
        self.word_freq_dict = {}
        self.indexed_word_dict = {} #dict that maps a unique index of a word to a list of its neighboring words
        # { i : w_i-2, w_i-1, w_i, w_i+1, w_i+2}
        self.indexed_tag_dict = {}
            
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
                    if ',' in word:
                        word = word.replace(',', 'comma')
                        tag = tag.replace(',', 'comma')
                    if ':' in word:
                        word = word.replace(':', 'colon')
                        tag = tag.replace(':', 'colon')
                    if '#' in word:
                        word = word.replace('#', 'hash')
                        tag = tag.replace('#', 'hash')
                    if ':' in tag:
                        tag = tag.replace(':', 'colon')

                    
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
                tag_minus_1 = "BOS" 
                word_minus_1 = "</s>"

            try:
                tag_minus_2 = list_of_word_tag_tuples[i-2][1]
                word_minus_2 = list_of_word_tag_tuples[i-2][0]
            except IndexError:
                tag_minus_2 = None 
                word_minus_2 = None

            try:
                word_plus_1 = list_of_word_tag_tuples[i+1][0]
            except IndexError:
                word_plus_1 = "<s>" 

            try:
                word_plus_2 = list_of_word_tag_tuples[i+2][0]
            except IndexError:
                word_plus_2 = None

            tags_list = [tag_minus_2, tag_minus_1, tag]
            words_list = [word_minus_2, word_minus_1, word, word_plus_1, word_plus_2]

            self.indexed_tag_dict[i] = tags_list
            self.indexed_word_dict[i] = words_list



        #### Sort the dictionary in a list of tuples with word, freq
        sorted_word_freq_dict = sorted(self.word_freq_dict.items(), key=lambda x: x[1], reverse=True)

        output_file_path = self.output_dir + "/train_voc"

        #### Print sorted word, freq to train_voc file
        with open(output_file_path, 'w') as file:
            for element, count in sorted_word_freq_dict:
                file.write(f"{element}\t{count}\n")

    def create_test_vectors(self):

        ##### READ IN TEST DATA ####

        list_of_word_tag_tuples = []

        with open(self.test_file, 'r') as file:
            for line in file:
                words_tags = line.rstrip().split()

                for element in words_tags:
                    try:
                        word, tag = element.split('/')
                    # This is so words with \/ are handled correctly. 
                    except ValueError:
                        word1, word2, tag = element.split('/')
                        word = word1 + word2

                    if ',' in word:
                        word = word.replace(',', 'comma')
                        tag = tag.replace(',', 'comma')
                    if ':' in word:
                        word = word.replace(':', 'colon')
                        tag = tag.replace(':', 'colon')
                    if '#' in word:
                        word = word.replace('#', 'hash')
                        tag = tag.replace('#', 'hash')
                    if ':' in tag:
                        tag = tag.replace(':', 'colon')
                    
                    word_tag_tuple = (word, tag)
                    list_of_word_tag_tuples.append(word_tag_tuple)

        ##### CREATE AND POPULATE index_word_dict AND tag_word_dict FOR TEST DATA #####

        self.test_indexed_word_dict = {}
        self.test_indexed_tag_dict = {}

        for i in range(len(list_of_word_tag_tuples)):
            word = list_of_word_tag_tuples[i][0]
            tag = list_of_word_tag_tuples[i][1]

            try:
                tag_minus_1 = list_of_word_tag_tuples[i-1][1]
                word_minus_1 = list_of_word_tag_tuples[i-1][0]
            except IndexError:
                tag_minus_1 = "BOS" 
                word_minus_1 = "</s>"

            try:
                tag_minus_2 = list_of_word_tag_tuples[i-2][1]
                word_minus_2 = list_of_word_tag_tuples[i-2][0]
            except IndexError:
                tag_minus_2 = None 
                word_minus_2 = None

            try:
                word_plus_1 = list_of_word_tag_tuples[i+1][0]
            except IndexError:
                word_plus_1 = "<s>" 

            try:
                word_plus_2 = list_of_word_tag_tuples[i+2][0]
            except IndexError:
                word_plus_2 = None

            tags_list = [tag_minus_2, tag_minus_1, tag]
            words_list = [word_minus_2, word_minus_1, word, word_plus_1, word_plus_2]

            self.test_indexed_tag_dict[i] = tags_list
            self.test_indexed_word_dict[i] = words_list

        ##### CREATE AND POPULATE A DICT OF FEATURES ####

        self.test_index_to_feature_list = {}

        for i in range(len(list_of_word_tag_tuples)):
            self.test_index_to_feature_list[i] = [] #for each token, store a list of its features.

            prevT = self.test_indexed_tag_dict[i][1]
            prevT_key = "prevT="+prevT
            self.test_index_to_feature_list[i].append(prevT_key+":1")

            prev2T = self.test_indexed_tag_dict[i][0]
            if prev2T != None:
                prevTwoTags_key = "prevTwoTags="+prev2T+ "+" +prevT 
                self.test_index_to_feature_list[i].append(prevTwoTags_key+":1")

            prevW = self.test_indexed_word_dict[i][1]
            prevW_key = "prevW="+prevW  
            self.test_index_to_feature_list[i].append(prevW_key+":1")

            prev2W = self.test_indexed_word_dict[i][0]
            if prev2W != None:
                prev2W_key = "prev2W="+prev2W  
                self.test_index_to_feature_list[i].append(prev2W_key+":1")

            nextW = self.test_indexed_word_dict[i][3]
            nextW_key = "nextW="+nextW
            self.test_index_to_feature_list[i].append(nextW_key+":1")

            next2W = self.test_indexed_word_dict[i][3]
            if next2W != None:
                next2W_key = "next2W="+next2W
                self.test_index_to_feature_list[i].append(next2W_key+":1")

            word = self.test_indexed_word_dict[i][2]
            word_isRare = self.isRare(word)

            if not word_isRare: # if word is not rare:
                curW_key = "curW="+word
                self.test_index_to_feature_list[i].append(curW_key+":1")

            else: # if word is rare:
                word_containsnum = containsNumber(word)
                word_containsUC = containsUpper(word)
                word_containsHyp = containsHyphen(word)

                if word_containsnum: 
                    self.test_index_to_feature_list[i].append("containNum:1")

                if word_containsUC:
                    self.test_index_to_feature_list[i].append("containUC:1")

                if word_containsHyp: 
                    self.test_index_to_feature_list[i].append("containHyp:1")

                ### prefix and suffix features ### 
                for j in range(len(word)):
                    if j == 0:
                        
                        pre = word[j]
                        pre_key = "pref=" + pre 
                        self.test_index_to_feature_list[i].append(pre_key+":1")

                        suf = word[-1]
                        suf_key = "suf=" + suf
                        self.test_index_to_feature_list[i].append(suf_key+":1")

                    elif (j == 1) or (j == 2) or (j == 3):
                        pre = word[:j+1]
                        pre_key = "pref=" + pre 
                        self.test_index_to_feature_list[i].append(pre_key+":1")

                        suf = word[(-j-1):]
                        suf_key = "suf=" + suf
                        self.test_index_to_feature_list[i].append(pre_key+":1")

   


    def create_init_feats(self):
        # create dicts storing each feature
        self.index_to_feature_list = {}


        #create dicts storing the count of each feature
        self.init_feat_freqs = defaultdict(int) #dict that maps the name of each feature to its frequency
        

        # loop through each token in the training file
        for i in self.indexed_word_dict:

            self.index_to_feature_list[i] = [] #for each token, store a list of its features.

            #first, do features that are going to be included regardless of whether or not w_i is rare

            prevT = self.indexed_tag_dict[i][1]
            prevT_key = "prevT="+prevT
            self.init_feat_freqs[prevT_key] += 1
            self.index_to_feature_list[i].append(prevT_key+":1")

            prev2T = self.indexed_tag_dict[i][0]
            if prev2T != None:
                prevTwoTags_key = "prevTwoTags="+prev2T+ "+" +prevT 
                self.init_feat_freqs[prevTwoTags_key] += 1
                self.index_to_feature_list[i].append(prevTwoTags_key+":1")

            prevW = self.indexed_word_dict[i][1]
            prevW_key = "prevW="+prevW  
            self.init_feat_freqs[prevW_key] += 1
            self.index_to_feature_list[i].append(prevW_key+":1")

            prev2W = self.indexed_word_dict[i][0]
            if prev2W != None:
                prev2W_key = "prev2W="+prev2W  
                self.init_feat_freqs[prev2W_key] += 1
                self.index_to_feature_list[i].append(prev2W_key+":1")

            nextW = self.indexed_word_dict[i][3]
            nextW_key = "nextW="+nextW
            self.init_feat_freqs[nextW_key] += 1 
            self.index_to_feature_list[i].append(nextW_key+":1")

            next2W = self.indexed_word_dict[i][3]
            if next2W != None:
                next2W_key = "next2W="+next2W
                self.init_feat_freqs[next2W_key] += 1 
                self.index_to_feature_list[i].append(next2W_key+":1")


            word = self.indexed_word_dict[i][2]
            word_isRare = self.isRare(word)

            if not word_isRare: # if word is not rare:

                curW_key = "curW="+word
                self.init_feat_freqs[curW_key] += 1
                self.index_to_feature_list[i].append(curW_key+":1")

            else: # if word is rare:

                word_containsnum = containsNumber(word)
                word_containsUC = containsUpper(word)
                word_containsHyp = containsHyphen(word)

                if word_containsnum: 
                    self.init_feat_freqs["containNum"] += 1
                    self.index_to_feature_list[i].append("containNum:1")

                if word_containsUC:
                    self.init_feat_freqs["containUC"] += 1
                    self.index_to_feature_list[i].append("containUC:1")

                if word_containsHyp: 
                    self.init_feat_freqs["containHyp"] += 1
                    self.index_to_feature_list[i].append("containHyp:1")

                ### prefix and suffix features ### 

                for j in range(len(word)):
                    if j == 0:
                        
                        pre = word[j]
                        pre_key = "pref=" + pre 
                        self.init_feat_freqs[pre_key] += 1
                        self.index_to_feature_list[i].append(pre_key+":1")

                        suf = word[-1]
                        suf_key = "suf=" + suf
                        self.init_feat_freqs[suf_key] += 1
                        self.index_to_feature_list[i].append(suf_key+":1")

                        
                        

                    elif (j == 1) or (j == 2) or (j == 3):
                        
                        pre = word[:j+1]
                        pre_key = "pref=" + pre 
                        self.init_feat_freqs[pre_key] += 1
                        self.index_to_feature_list[i].append(pre_key+":1")

                        suf = word[(-j-1):]
                        suf_key = "suf=" + suf
                        self.init_feat_freqs[suf_key] += 1
                        self.index_to_feature_list[i].append(pre_key+":1")

                        

                        

    def create_kept_feats(self):

        self.kept_feat_freqs = defaultdict(int) #same as init_feat_freqs, but only counts the features that are kept after applying feat_thres.

        for feat in self.init_feat_freqs:

            #keep all w_i features, regardless of frequency
            if "curW=" in feat:
                self.kept_feat_freqs[feat] = self.init_feat_freqs[feat]

            else:
                # for all other features:

                if self.init_feat_freqs[feat] >= self.feat_thres: # if a feat appears less than feat_thres, don't add it
                    self.kept_feat_freqs[feat] = self.init_feat_freqs[feat]


    def print_kept_feats(self):
        sorted_kept_feats = sorted(self.kept_feat_freqs.items(), key=lambda x: x[1], reverse=True)
        file_path = self.output_dir + "/kept_feats"
        with open(file_path, 'w') as outfile:
            for key, item in sorted_kept_feats:
                outfile.write(key + " " + str(item) +"\n")
            
                
    def print_init_feats(self):
        sorted_init_feats = sorted(self.init_feat_freqs.items(), key=lambda x: x[1], reverse=True)
        file_path = self.output_dir + "/init_feats"
        with open(file_path, 'w') as outfile:
            for key, item in sorted_init_feats:
                outfile.write(key + " " + str(item) +"\n")

    def print_train_vectors(self):

        file_path = self.output_dir + "/final_train.vectors.txt"

        with open(file_path, 'w') as outfile:

            for i in self.index_to_feature_list:
                tag = self.indexed_tag_dict[i][2]
                toprint = tag 

                for feature in self.index_to_feature_list[i]:
                    if feature[:-2] in self.kept_feat_freqs: # for this file, we only keep features in kept_freqs
                        toprint += " " + feature 

                toprint += "\n"
                outfile.write(toprint)

    def print_test_vectors(self):
        file_path = self.output_dir + "/final_test.vectors.txt"

        with open(file_path, 'w') as outfile:

            for i in self.test_index_to_feature_list:
                tag = self.test_indexed_tag_dict[i][2]
                toprint = tag 

                for feature in self.test_index_to_feature_list[i]:
                    if feature[:-2] in self.kept_feat_freqs: # only keep features in kept_freqs which we got from the training data
                        toprint += " " + feature 

                toprint += "\n"
                outfile.write(toprint)



    def isRare(self, word):
        if word not in self.word_freq_dict: #if word doesnt appear at all, then it's definitely rare 
            return True

        if self.word_freq_dict[word] < self.rare_thres:
            return True
        else:
            return False 



#######FEATURE HELPER FUNCS#######

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
    tagger.create_init_feats()
    tagger.print_init_feats()
    tagger.create_kept_feats()
    tagger.print_kept_feats()
    tagger.print_train_vectors()

    tagger.create_test_vectors()
    tagger.print_test_vectors()


if __name__ == "__main__":
    main()

####################### Debugging, delete later ##########################

#print(sorted_word_freq_dict)
#print(word_freq_dict)
#print(list_of_word_tag_tuples)

        #print(words_tags)
        #print(line)


