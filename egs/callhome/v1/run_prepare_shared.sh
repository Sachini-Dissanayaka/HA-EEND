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
    echo "prepare kaldi-style datasets"
#     mkdir -p data/local
    local/download_and_untar.sh data/local dev-clean
    local/download_and_untar.sh data/local test-clean
    local/download_and_untar.sh data/local train-clean-100

    if [ ! -f data/dev_clean/.done ]; then
        local/data_prep.sh data/local/LibriSpeech/dev-clean data/dev_clean || exit
        touch data/dev_clean/.done
    fi
    if [ ! -f data/test_clean/.done ]; then
        local/data_prep.sh data/local/LibriSpeech/test-clean data/test_clean || exit
        touch data/test_clean/.done
    fi
    if [ ! -f data/train_clean_100/.done ]; then    
        local/data_prep.sh data/local/LibriSpeech/train-clean-100 data/train_clean_100
        touch data/train_clean_100/.done
    fi

    if [ ! -d data/musan_bgnoise ]; then
        tar xzf musan_bgnoise.tar.gz
    fi
    if [ ! -f data/simu_rirs_8k/.done ]; then
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
        touch data/simu_rirs_8k/.done
    fi
fi

simudir=data/simu
if [ $stage -le 2 ]; then
    echo "simulation of mixture"
    mkdir -p $simudir/.work
    random_mixture_cmd=random_mixture_nooverlap.py
    make_mixture_cmd=make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=random_mixture.py
        make_mixture_cmd=make_mixture.py
    fi

    for simu_opts_sil_scale in 2; do
        for dset in train_clean_100 dev_clean test_clean; do
            if [ "$dset" == "train_clean_100" ]; then
                n_mixtures=2000
            else
                n_mixtures=500
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $simu_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    data/$dset data/musan_bgnoise data/simu_rirs_8k \
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

if [ $stage -le 3 ]; then
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
