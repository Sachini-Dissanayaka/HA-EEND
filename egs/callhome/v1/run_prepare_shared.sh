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

callhome_dir=$PWD/data/local/nist_recognition_evaluation

# Modify simulated data storage area.
# This script distributes simulated data under these directories
simu_actual_dirs=(
$PWD/data/local/diarization-data
)

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker=2
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=100
simu_opts_min_utts=10
simu_opts_max_utts=20

. path.sh
. cmd.sh
. parse_options.sh || exit

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
    # Prepare CALLHOME dataset. This will be used to evaluation.
    if ! validate_data_dir.sh --no-text --no-feats data/callhome1_spk2 \
        || ! validate_data_dir.sh --no-text --no-feats data/callhome2_spk2; then
        # imported from https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v1
        local/make_callhome.sh $callhome_dir data
        # Generate two-speaker subsets
        for dset in callhome1 callhome2; do
            # Extract two-speaker recordings in wav.scp
            copy_data_dir.sh data/${dset} data/${dset}_spk2
            utils/filter_scp.pl <(awk '{if($2==2) print;}'  data/${dset}/reco2num_spk) \
                data/${dset}/wav.scp > data/${dset}_spk2/wav.scp
            # Regenerate segments file from fullref.rttm
            #  $2: recid, $4: start_time, $5: duration, $8: speakerid
            awk '{printf "%s_%s_%07d_%07d %s %.2f %.2f\n", \
                 $2, $8, $4*100, ($4+$5)*100, $2, $4, $4+$5}' \
                data/callhome/fullref.rttm | sort > data/${dset}_spk2/segments
            utils/fix_data_dir.sh data/${dset}_spk2
            # Speaker ID is '[recid]_[speakerid]
            awk '{split($1,A,"_"); printf "%s %s_%s\n", $1, A[1], A[2]}' \
                data/${dset}_spk2/segments > data/${dset}_spk2/utt2spk
            utils/fix_data_dir.sh data/${dset}_spk2
            # Generate rttm files for scoring
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                data/${dset}_spk2/utt2spk data/${dset}_spk2/segments \
                data/${dset}_spk2/rttm
            utils/data/get_reco2dur.sh data/${dset}_spk2
        done
    fi 
fi

if [ $stage -le 1 ]; then
    # compose eval/callhome2_spk2
    eval_set=data/eval/callhome2_spk2
    if ! validate_data_dir.sh --no-text --no-feats $eval_set; then
        utils/copy_data_dir.sh data/callhome2_spk2 $eval_set
        cp data/callhome2_spk2/rttm $eval_set/rttm
        awk -v dstdir=wav/eval/callhome2_spk2 '{print $1, dstdir"/"$1".wav"}' data/callhome2_spk2/wav.scp > $eval_set/wav.scp
        mkdir -p wav/eval/callhome2_spk2
        wav-copy scp:data/callhome2_spk2/wav.scp scp:$eval_set/wav.scp
        utils/data/get_reco2dur.sh $eval_set
    fi

    # compose eval/callhome1_spk2
    adapt_set=data/eval/callhome1_spk2
    if ! validate_data_dir.sh --no-text --no-feats $adapt_set; then
        utils/copy_data_dir.sh data/callhome1_spk2 $adapt_set
        cp data/callhome1_spk2/rttm $adapt_set/rttm
        awk -v dstdir=wav/eval/callhome1_spk2 '{print $1, dstdir"/"$1".wav"}' data/callhome1_spk2/wav.scp > $adapt_set/wav.scp
        mkdir -p wav/eval/callhome1_spk2
        wav-copy scp:data/callhome1_spk2/wav.scp scp:$adapt_set/wav.scp
        utils/data/get_reco2dur.sh $adapt_set
    fi
fi
