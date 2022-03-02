#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options

stage=0

# Modify corpus directories
#  - callhome_dir
#    CALLHOME (LDC2001S97)
#  - swb2_phase1_train
#    Switchboard-2 Phase 1 (LDC98S75)
#  - data_root
#    LDC99S79, LDC2002S06, LDC2001S13, LDC2004S07,
#    LDC2006S44, LDC2011S01, LDC2011S04, LDC2011S09,
#    LDC2011S10, LDC2012S01, LDC2011S05, LDC2011S08
#  - musan_root
#    MUSAN corpus (https://www.openslr.org/17/)
# callhome_dir=/export/corpora/NIST/LDC2001S97
data_root=$PWD/data/local/SinhalaASR
sinhala_asr_dir=$PWD/data/sinhala_asr_train
musan_root=$PWD/data/musan_root
# Modify simulated data storage area.
# This script distributes simulated data under these directories
simu_actual_dirs=(
$PWD/data/local/diarization-data
)

# data preparation options
max_jobs_run=4
sad_num_jobs=30
sad_opts="--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3"
sad_graph_opts="--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0"
sad_priors_opts="--sil-scale=0.1"

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker=2
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=5000
simu_opts_min_utts=10
simu_opts_max_utts=20

. path.sh
. cmd.sh
. parse_options.sh || exit

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"

    # Prepare a collection of Sinhala ASR data. This will be used to train,
    if ! validate_data_dir.sh --no-text --no-feats data/sinhala_asr_train; then
        local/make_sinhala_asr.sh $data_root 
    fi

    if [ ! -d data/musan_root ]; then
        # local/download_musan.sh
        tar xzf musan.tar.gz
        # rm -f "musan.tar.gz"
    fi

    # musan data. "back-ground
    if ! validate_data_dir.sh --no-text --no-feats data/musan_noise_bg; then
        local/make_musan.sh $musan_root data
        utils/copy_data_dir.sh data/musan_noise data/musan_noise_bg
        awk '{if(NR>1) print $1,$1}'  $musan_root/noise/free-sound/ANNOTATIONS > data/musan_noise_bg/utt2spk
        utils/fix_data_dir.sh data/musan_noise_bg
    fi

    # simu rirs 8k
    if ! validate_data_dir.sh --no-text --no-feats data/simu_rirs_8k; then
        mkdir -p data/simu_rirs_8k
        if [ ! -e sim_rir_8k.zip ]; then
            wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_8k.zip
        fi
        unzip sim_rir_8k.zip -d data/sim_rir_8k
        find $PWD/data/sim_rir_8k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/simu_rirs_8k/wav.scp
        awk '{print $1, $1}' data/simu_rirs_8k/wav.scp > data/simu_rirs_8k/utt2spk
        utils/fix_data_dir.sh data/simu_rirs_8k
    fi
    # Automatic segmentation using pretrained SAD model
    #     it will take one day using 30 CPU jobs:
    #     make_mfcc: 1 hour, compute_output: 18 hours, decode: 0.5 hours
    sad_nnet_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
    sad_work_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
    if ! validate_data_dir.sh --no-text $sad_work_dir/sinhala_asr_dir_seg; then
        if [ ! -d exp/segmentation_1a ]; then
            wget http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
            tar zxf 0004_tdnn_stats_asr_sad_1a.tar.gz
        fi
        steps/segmentation/detect_speech_activity.sh \
            --nj $sad_num_jobs \
            --graph-opts "$sad_graph_opts" \
            --transform-probs-opts "$sad_priors_opts" $sad_opts \
            $sinhala_asr_dir $sad_nnet_dir mfcc_hires $sad_work_dir \
            $sad_work_dir/sinhala_asr_dir || exit 1
    fi
    # Extract >1.5 sec segments and split into train/valid sets
    if ! validate_data_dir.sh --no-text --no-feats data/sinhala_asr_cv; then
        copy_data_dir.sh $sinhala_asr_dir data/sinhala_asr_dir_seg
        awk '$4-$3>1.5{print;}' $sad_work_dir/sinhala_asr_dir_seg/segments > data/sinhala_asr_dir_seg/segments
        cp $sad_work_dir/sinhala_asr_dir_seg/{utt2spk,spk2utt} data/sinhala_asr_dir_seg
        utils/fix_data_dir.sh data/sinhala_asr_dir_seg
        local/subset_data_dir_tr_dev.sh data/sinhala_asr_dir_seg data/sinhala_asr_tr data/sinhala_asr_dev
        local/subset_data_dir_dev_test.sh data/sinhala_asr_dev data/sinhala_asr_cv data/sinhala_asr_test
    fi
fi

simudir=data/simu
if [ $stage -le 1 ]; then
    echo "simulation of mixture"
    mkdir -p $simudir/.work
    random_mixture_cmd=random_mixture_nooverlap.py
    make_mixture_cmd=make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=random_mixture.py
        make_mixture_cmd=make_mixture.py
    fi

    for simu_opts_sil_scale in 2; do
        for dset in sinhala_asr_tr sinhala_asr_cv sinhala_asr_test; do
            if [ "$dset" == "sinhala_asr_tr" ]; then
                n_mixtures=${simu_opts_num_train}
            else
                n_mixtures=500
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $train_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    data/$dset data/musan_noise_bg data/simu_rirs_8k \
                    \> $simudir/.work/mixture_$simuid.scp
                nj=100
                mkdir -p $simudir/wav/$simuid
                # distribute simulated data to $simu_actual_dir
                split_scps=
                for n in $(seq $nj); do
                    split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
                    mkdir -p $simudir/.work/data_$simuid.$n
                    actual=${simu_actual_dirs[($n-1)%${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
                    mkdir -p $actual
                    ln -nfs $actual $simudir/wav/$simuid/$n
                done
                utils/split_scp.pl $simudir/.work/mixture_$simuid.scp $split_scps || exit 1

                $simu_cmd --max-jobs-run 32 JOB=1:$nj $simudir/.work/make_mixture_$simuid.JOB.log \
                    $make_mixture_cmd --rate=8000 \
                    $simudir/.work/mixture_$simuid.JOB.scp \
                    $simudir/.work/data_$simuid.JOB $simudir/wav/$simuid/JOB
                utils/combine_data.sh $simudir/data/$simuid $simudir/.work/data_$simuid.*
                steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    $simudir/data/$simuid/utt2spk $simudir/data/$simuid/segments \
                    $simudir/data/$simuid/rttm
                utils/data/get_reco2dur.sh $simudir/data/$simuid
            fi
        done
    done
fi
