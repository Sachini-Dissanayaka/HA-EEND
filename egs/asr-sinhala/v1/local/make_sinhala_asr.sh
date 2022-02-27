#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.

# for d in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
#   resource="52/asr_sinhala_${d}.zip"
#   wget "http://www.openslr.org/resources/$resource"
#   zipfile="$(basename "$resource")"
#   unzip -nqq "$zipfile"
#   rm -f "$zipfile"
# done

if [ ! -d "$KALDI_ROOT" ] ; then
  echo >&2 'KALDI_ROOT must be set and point to the Kaldi directory'
  exit 1
fi

set -o errexit
set -o nounset
export LC_ALL=C

readonly CORPUSDIR="$1"

#
# Kaldi recipe directory layout
#

# Create the directories needed
mkdir -p data/sinhala_asr_train

#
# Train data
#

full_file=data/local/utt_spk_text.tsv

echo "Preparing sinhala asr data, this may take a while"
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --alsent > data/sinhala_asr_train/al_sent.txt
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --spk2utt    | sort -k1,1 > data/sinhala_asr_train/spk2utt
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --spk2gender | sort -k1,1 > data/sinhala_asr_train/spk2gender
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --text       | sort -k1,1 > data/sinhala_asr_train/text
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --utt2spk    | sort -k1,1 > data/sinhala_asr_train/utt2spk
local/kaldi_converter.py -d $CORPUSDIR -f $full_file --wavscp     | sort -k1,1 > data/sinhala_asr_train/wav.scp
echo "Sinhala ASR data prepared"

# Fix sorting issues etc.
utils/fix_data_dir.sh data/sinhala_asr_train