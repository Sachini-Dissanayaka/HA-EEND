#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

exp_dir=exp/diarize
conf_dir=conf

# train_id=$(basename $train_set)
# valid_id=$(basename $valid_set)
# train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
# infer_config_id=$(echo $infer_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')

# # Additional arguments are added to config_id
# train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
# infer_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

# model_id=$train_id.$valid_id.$train_config_id
# model_dir=exp/diarize/model/$model_id
model_dir=exp/diarize/model

train_dir=data/simu/data/train_clean_100_ns2_beta2_2000
dev_dir=data/simu/data/dev_clean_ns2_beta2_2000
test_dir=$PWD/data/simu/data/test_clean_ns2_beta2_500
train_conf=$conf_dir/train.yaml

init_model=$model_dir/avg.th

infer_conf=$conf_dir/infer.yaml
infer_out_dir=$exp_dir/infer/simu
# test_dir=data/simu/data/swb_sre_cv_ns2_beta2_500
test_model=$model_dir/avg.th
# infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/librispeech/.work 
scoring_dir=$exp_dir/scoring/librispeech
# scoring_dir=$exp_dir/score/simu

stage=1

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python /home/sachini/HA-EEND/eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir

fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python /home/sachini/HA-EEND/eend/bin/model_averaging.py $init_model $ifiles

fi

# Adapting
# if [ $stage -le 3 ]; then
#     echo "Start adapting"
#     python eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model
# fi

# Model averaging
# if [ $stage -le 3 ]; then
#     echo "Start model averaging"
#     ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
#     python eend/bin/model_averaging.py $test_model $ifiles
# fi

# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
     python /home/sachini/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	 python /home/sachini/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
		$work/file_list $scoring_dir/hyp_${th}_$med.rttm
	md-eval.pl -c 0.25 \
		-r $test_dir/rttm \
		-s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

if [ $stage -le 5 ]; then
    best_score.sh $scoring_dir
fi
echo "Finished !"