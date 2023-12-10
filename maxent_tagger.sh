#maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir

#!/bin/sh

# DO NOT change the path of the interpreter in your homework
/nopt/dropbox/23-24/570/envs/570/bin/python maxent_tagger.py "$1" "$2" "$3" "$4" "$5"

mallet import-svmlight --input "$5"/final_train.vectors.txt --output "$5"/final_train.vectors

mallet import-svmlight --input "$5"/final_test.vectors.txt --output "$5"/final_test.vectors --use-pipe-from "$5"/final_train.vectors

eval mallet train-classifier --trainer MaxEnt --input "$5"/final_train.vectors --output-classifier "$5"/me_model > "$5"/me_model.stdout 2>"$5"/me_model.stderr

classifier2info --classifier me_model > me_model.txt

mallet classify-svmlight --input "$5"/final_test.vectors.txt --classifier "$5"/me_model --output "$5"/sys_out 


vectors2classify --training-file "$5"/final_train.vectors --testing-file "$5"/final_test.vectors --trainer MaxEnt --report test:raw test:accuracy train:accuracy 
