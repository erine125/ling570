#maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir

#!/bin/sh

# DO NOT change the path of the interpreter in your homework
/nopt/dropbox/23-24/570/envs/570/bin/python maxent_tagger.py "$1" "$2" "$3" "$4" "$5"

mallet import-svmlight --input "$5"/final_train.vectors.txt --output "$5"/final_train.vectors

mallet import-svmlight --input "$5"/final_test.vectors.txt --output "$5"/final_test.vectors --use-pipe-from "$5"/final_train.vectors

vectors2classify --training-file "$5"/final_train.vectors --testing-file "$5"/final_test.vectors --trainer MaxEnt 
