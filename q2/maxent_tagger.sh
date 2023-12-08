#maxent_tagger.sh train_file test_file rare_thres feat_thres output_dir

#!/bin/sh

# DO NOT change the path of the interpreter in your homework
/nopt/dropbox/23-24/570/envs/570/bin/python maxent_tagger.py "$1" "$2" "$3" "$4" "$5"
