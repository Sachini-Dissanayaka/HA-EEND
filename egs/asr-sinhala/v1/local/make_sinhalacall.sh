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

if [ ! -d "$1" ] ; then
  echo >&2 "Usage: prep.sh CALLSINHALA1"
  exit 1
fi

if [ ! -d "$2" ] ; then
  echo >&2 "Usage: prep.sh CALLSINHALA2"
  exit 1
fi

if [ ! -d "$KALDI_ROOT" ] ; then
  echo >&2 'KALDI_ROOT must be set and point to the Kaldi directory'
  exit 1
fi

set -o errexit
set -o nounset
export LC_ALL=C

# readonly CORPUSDIR="$1"
readonly CALLSINHALA1="$1"
readonly CALLSINHALA2="$2"

#
# Kaldi recipe directory layout
#

# Create the directories needed
mkdir -p data/callsinhala1 data/callsinhala2

#
# Test and adapt data
#

sinhala_file1=data/local/CallSinhala/callsinhala1/utt_spk_seg.tsv
sinhala_file2=data/local/CallSinhala/callsinhala2/utt_spk_seg.tsv

echo "Preparing callsinhala1 data, this may take a while"
local/kaldi_converter.py -d $CALLSINHALA1 -f $sinhala_file1 --spk2utt    | sort -k1,1 > data/callsinhala1/spk2utt
local/kaldi_converter.py -d $CALLSINHALA1 -f $sinhala_file1 --utt2spk    | sort -k1,1 > data/callsinhala1/utt2spk
local/kaldi_converter.py -d $CALLSINHALA1 -f $sinhala_file1 --wavscp     | sort -k1,1 > data/callsinhala1/wav.scp
local/kaldi_converter.py -d $CALLSINHALA1 -f $sinhala_file1 --segments     | sort -k1,1 > data/callsinhala1/segments
echo "callsinhala1 data prepared"

echo "Preparing callsinhala2 data, this may take a while"
local/kaldi_converter.py -d $CALLSINHALA2 -f $sinhala_file2 --spk2utt    | sort -k1,1 > data/callsinhala2/spk2utt
local/kaldi_converter.py -d $CALLSINHALA2 -f $sinhala_file2 --utt2spk    | sort -k1,1 > data/callsinhala2/utt2spk
local/kaldi_converter.py -d $CALLSINHALA2 -f $sinhala_file2 --wavscp     | sort -k1,1 > data/callsinhala2/wav.scp
local/kaldi_converter.py -d $CALLSINHALA2 -f $sinhala_file2 --segments     | sort -k1,1 > data/callsinhala2/segments
echo "callsinhala2 data prepared"

# Fix sorting issues etc.
utils/fix_data_dir.sh data/callsinhala1
utils/fix_data_dir.sh data/callsinhala2