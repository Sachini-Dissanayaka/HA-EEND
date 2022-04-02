#! /usr/bin/env python

# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to convert open-source speech corpora to Kaldi RM recipe format."""

from __future__ import unicode_literals

import io
from operator import itemgetter
import optparse
import os.path
import corpus_util as kaldi_util

stdin = io.open(0, mode='rt', encoding='utf-8', closefd=False)
stdout = io.open(1, mode='wt', encoding='utf-8', closefd=False)
stderr = io.open(2, mode='wt', encoding='utf-8', closefd=False)


class CorpusConverter(object):
  """Container for cretion from open-soruce corpus to RM recipe."""

  def __init__(self, corpus):
    self.corpus = corpus
    self.corpus_info = corpus.corpus_info.items() # {utterance_id:row}

  def Spk2utt(self):
    """Prints out the text used in spk2utt file in RM recipe."""
    spk_utt = {}
    for _, rec in self.corpus_info:
      if rec.session_id not in spk_utt:
        spk_utt[rec.session_id] = []
      spk_utt[rec.session_id].append(rec.utterance_id)

    for session_id in spk_utt:
      stdout.write('%s' % session_id)
      for utt_id in spk_utt[session_id]:
        stdout.write(' %s' % utt_id)
      stdout.write('\n')

  def Utt2spk(self):
    """Prints out the text used in utt2spk file in RM recipe."""
    spk_utt = {}
    for _, rec in self.corpus_info:
      if rec.session_id not in spk_utt:
        spk_utt[rec.session_id] = []
      spk_utt[rec.session_id].append(rec.utterance_id)

    for session_id in spk_utt:
      for utt_id in spk_utt[session_id]:
        stdout.write('%s %s\n' % (utt_id, session_id))

  def Wavscp(self):
    """Prints out the text used in wav.scp file in RM recipe."""
    recordings = []

    for _, rec in self.corpus_info:
      if rec.recording_id not in recordings:
        recordings.append(rec.recording_id) 
        path = os.path.join(self.corpus.corpus_dir,
                            '%s.flac' % rec.recording_id)
        stdout.write('%s flac -cds %s |\n' % (rec.recording_id, path))
   
  def Segments(self):
    """Prints out the segments files"""
    for _, rec in self.corpus_info:
      stdout.write('%s %s %s %s\n' % (rec.utterance_id, rec.recording_id, rec.segment_begin, rec.segment_end))


def main():
  parser = optparse.OptionParser()
  parser.add_option('-d',
                    '--dir',
                    dest='corpusdir',
                    help='Input corpus directory')
  parser.add_option('--spk2utt',
                    dest='spk2utt',
                    action='store_false',
                    help='Output for spk2utt file')
  parser.add_option('--utt2spk',
                    dest='utt2spk',
                    action='store_false',
                    help='Output for utt2spk file')
  parser.add_option('--wavscp',
                    dest='wavscp',
                    action='store_false',
                    help='Output for wac.scp file')
  parser.add_option('--segments',
                    dest='segments',
                    action='store_false',
                    help='Output for segments file')
  parser.add_option('-f',
                    '--file',
                    dest='corpusfile',
                    help='Output training data')
  parser.add_option('--testfile',
                    dest='corpus_test_file',
                    help='Output testfile data')

  options, _ = parser.parse_args()

  corpus = kaldi_util.Corpus(options.corpusdir, options.corpusfile)
  corpus.LoadItems()
  kaldi_converter = CorpusConverter(corpus)


  if options.spk2utt is not None:
    kaldi_converter.Spk2utt()

  if options.utt2spk is not None:
    kaldi_converter.Utt2spk()

  if options.wavscp is not None:
    kaldi_converter.Wavscp()

  if options.segments is not None:
    kaldi_converter.Segments()


if __name__ == '__main__':
  main()
