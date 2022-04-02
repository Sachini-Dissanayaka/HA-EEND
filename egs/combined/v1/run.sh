#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

exp_dir=exp/diarize
conf_dir=conf

model_dir=$exp_dir/model

train_dir=data/simu/data/sinhala_asr_tr_ns2_beta2_5000
dev_dir=data/simu/data/sinhala_asr_cv_ns2_beta2_500
test_dir=data/simu/data/sinhala_asr_test_ns2_beta2_500
test_dir_cs=data/callsinhala2
test_dir_ch=data/eval/callhome2_spk2
train_conf=$conf_dir/train.yaml

train_adapt_dir_cs=data/callsinhala1
dev_adapt_dir_cs=data/callsinhala2
model_adapt_dir_cs=$exp_dir/models_adapt_cs
train_adapt_dir_ch=data/eval/callhome1_spk2
dev_adapt_dir_ch=data/eval/callhome2_spk2
model_adapt_dir_ch=$exp_dir/models_adapt_ch
adapt_conf=$conf_dir/adapt.yaml

init_model=$model_dir/avg.th

infer_conf=$conf_dir/infer.yaml
infer_out_dir=$exp_dir/infer/sinhala-asr
infer_out_dir_cs=$exp_dir/infer/callsinhala
infer_out_dir_ch=$exp_dir/infer/callhome

test_model=$model_dir/avg.th
test_model_cs=$model_adapt_dir/avg.th
test_model_ch=$model_adapt_dir/avg.th

work=$infer_out_dir/sinhala-asr/.work
work_cs=$infer_out_dir/callsinhala/.work
work_ch=$infer_out_dir/callhome/.work

scoring_dir=$exp_dir/score/sinhala-asr
scoring_dir_cs=$exp_dir/score/callsinhala
scoring_dir_ch=$exp_dir/score/callhome

stage=1

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python ~/HA-EEND/eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python ~/HA-EEND/eend/bin/model_averaging.py $init_model $ifiles
fi

# Adapting CALLSINHALA
if [ $stage -le 3 ]; then
    echo "Start adapting"
    python ~/HA-EEND/eend/bin/train.py -c $adapt_conf $train_adapt_dir_cs $dev_adapt_dir_cs $model_adapt_dir_cs --initmodel $init_model
fi

# Model averaging 
if [ $stage -le 3 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_adapt_dir_cs/transformer{91..100}.th`
    python ~/HA-EEND/eend/bin/model_averaging.py $test_model_cs $ifiles
fi

# Adapting CALLHOME
if [ $stage -le 3 ]; then
    echo "Start adapting"
    python ~/HA-EEND/eend/bin/train.py -c $adapt_conf $train_adapt_dir_ch $dev_adapt_dir_ch $model_adapt_dir_ch --initmodel $init_model
fi

# Model averaging 
if [ $stage -le 3 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_adapt_dir_ch/transformer{91..100}.th`
    python ~/HA-EEND/eend/bin/model_averaging.py $test_model_ch $ifiles
fi

# --- Sinhala ASR

Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
    python ~/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python ~/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
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

# --- CALLSINHALA

# Inferring
if [ $stage -le 6 ]; then
    echo "Start inferring"
    python ~/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir_cs $test_model_cs $infer_out_dir_cs
fi

# Scoring
if [ $stage -le 7 ]; then
    echo "Start scoring"
    mkdir -p $work_cs
    mkdir -p $scoring_dir_cs
	find $infer_out_dir_cs -iname "*.h5" > $work_cs/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python ~/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
		$work_cs/file_list $scoring_dir_cs/hyp_${th}_$med.rttm
	md-eval.pl -c 0.25 \
		-r $test_dir_cs/rttm \
		-s $scoring_dir_cs/hyp_${th}_$med.rttm > $scoring_dir_cs/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

if [ $stage -le 8 ]; then
    best_score.sh $scoring_dir_cs
fi

# --- CALLHOME

# Inferring
if [ $stage -le 6 ]; then
    echo "Start inferring"
    python ~/HA-EEND/eend/bin/infer.py -c $infer_conf $test_dir_ch $test_model_ch $infer_out_dir_ch
fi

# Scoring
if [ $stage -le 7 ]; then
    echo "Start scoring"
    mkdir -p $work_ch
    mkdir -p $scoring_dir_ch
	find $infer_out_dir_ch -iname "*.h5" > $work_ch/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python ~/HA-EEND/eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
		$work_ch/file_list $scoring_dir_ch/hyp_${th}_$med.rttm
	md-eval.pl -c 0.25 \
		-r $test_dir_ch/rttm \
		-s $scoring_dir_ch/hyp_${th}_$med.rttm > $scoring_dir_ch/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

if [ $stage -le 8 ]; then
    best_score.sh $scoring_dir_ch
fi

echo "Finished !"