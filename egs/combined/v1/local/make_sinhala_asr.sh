#! /bin/bash

# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script which takes an utt_spk_text.tsv file and generates Kaldi style files

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
mkdir -p data/sinhala_asr_full

#
# Complete data
#

full_file=data/local/utt_spk_text.tsv

echo "Preparing sinhala asr data, this may take a while"
# local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --alsent > data/sinhala_asr_full/al_sent.txt
local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --spk2utt    | sort -k1,1 > data/sinhala_asr_full/spk2utt
local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --spk2gender | sort -k1,1 > data/sinhala_asr_full/spk2gender
# local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --text       | sort -k1,1 > data/sinhala_asr_full/text
local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --utt2spk    | sort -k1,1 > data/sinhala_asr_full/utt2spk
local/kaldi_converter_1.py -d $CORPUSDIR -f $full_file --wavscp     | sort -k1,1 > data/sinhala_asr_full/wav.scp
echo "Sinhala ASR data prepared"

# Fix sorting issues etc.
utils/fix_data_dir.sh data/sinhala_asr_full