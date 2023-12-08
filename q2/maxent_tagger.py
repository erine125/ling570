# TERMINAL COMMAND: maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir
# *right now I am trying it with ./maxent_tagger.sh test.word_pos test_file rare_thres feat_thres output_dir

import sys

list_of_word_tag_tuples = []
word_freq_dict = {}

train_file = sys.argv[1]
test_file = sys.argv[2]
rare_thres = sys.argv[3]
feat_thres = sys.argv[4]
output_dir = sys.argv[5]


######################## Create train_voc with words and frequencies ###################################

#### Create a list of tuples with word, tag
with open(train_file, 'r') as file:
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
for word, tag in list_of_word_tag_tuples:
    if word in word_freq_dict:
        word_freq_dict[word] += 1
    else:
        word_freq_dict[word] = 1

#### Sort the dictionary in a list of tuples with word, freq
sorted_word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

#### Print sorted word, freq to train_voc file
# *We need to write this file in the output_dir folder. Right now it is just creating it in the current folder.
with open('train_voc', 'w') as file:
    for element, count in sorted_word_freq_dict:
        file.write(f"{element}\t{count}\n")




####################### Debugging, delete later ##########################

#print(sorted_word_freq_dict)
#print(word_freq_dict)
#print(list_of_word_tag_tuples)

        #print(words_tags)
        #print(line)


