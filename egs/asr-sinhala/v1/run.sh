#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

exp_dir=exp/diarize
conf_dir=conf

model_dir=$exp_dir/model

train_dir=data/simu/data/sinhala_asr_tr_ns2_beta2_2000
dev_dir=data/simu/data/sinhala_asr_cv_ns2_beta2_500
test_dir_ch=data/eval/callhome2_spk2
test_dir=data/simu/data/test_clean_ns2_beta2_500
train_conf=$conf_dir/train.yaml

train_adapt_dir=data/eval/callhome1_spk2
dev_adapt_dir=data/eval/callhome2_spk2
model_adapt_dir=$exp_dir/models_adapt
adapt_conf=$conf_dir/adapt.yaml

init_model=$model_dir/avg.th

infer_conf=$conf_dir/infer.yaml
infer_out_dir_ch=$exp_dir/infer/callhome
infer_out_dir=$exp_dir/infer/sinhala-asr

test_model=$model_dir/avg.th
test_model_ch=$model_adapt_dir/avg.th

work_ch=$infer_out_dir/callhome/.work
work=$infer_out_dir/sinhala-asr/.work
scoring_dir_ch=$exp_dir/score/callhome
scoring_dir=$exp_dir/score/sinhala-asr

stage=1

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python /home/yoshani/HA-EEND/eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python /home/yoshani/HA-EEND/eend/bin/model_averaging.py $init_model $ifiles
fi

# # Adapting
# if [ $stage -le 3 ]; then
#     echo "Start adapting"
#     python /home/yoshani/HA-EEND/eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model
# fi

# # Model averaging
# if [ $stage -le 3 ]; then
#     echo "Start model averaging"
#     ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
#     python /home/yoshani/HA-EEND/eend/bin/model_averaging.py $test_model_ch $ifiles
# fi

# --- Sinhala ASR

# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
    python /home/yoshani/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python /home/yoshani/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
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

# # --- CALLHOME

# # Inferring
# if [ $stage -le 6 ]; then
#     echo "Start inferring"
#     python /home/yoshani/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir_ch $test_model_ch $infer_out_dir_ch
# fi

# # Scoring
# if [ $stage -le 7 ]; then
#     echo "Start scoring"
#     mkdir -p $work_ch
#     mkdir -p $scoring_dir_ch
# 	find $infer_out_dir_ch -iname "*.h5" > $work_ch/file_list
# 	for med in 1 11; do
# 	for th in 0.3 0.4 0.5 0.6 0.7; do
# 	python /home/yoshani/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
# 		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
# 		$work_ch/file_list $scoring_dir_ch/hyp_${th}_$med.rttm
# 	md-eval.pl -c 0.25 \
# 		-r $test_dir_ch/rttm \
# 		-s $scoring_dir_ch/hyp_${th}_$med.rttm > $scoring_dir_ch/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
# 	done
# 	done
# fi

# if [ $stage -le 8 ]; then
#     best_score.sh $scoring_dir_ch
# fi

echo "Finished !"