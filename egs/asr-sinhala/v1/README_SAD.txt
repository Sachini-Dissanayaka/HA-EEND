 UPLOADER        Vimal Manohar
 DATE            2018-24-04
 KALDI VERSION   8ff7fd9

 This directory contains speech activity detection model from the recipe in 
 egs/aspire/s5.
 This was created when Kaldi's master branch was at git
 log 8ff7fd9f84e89a652716956d8989e9205d7bf52f.


 I. Files list
 ------------------------------------------------------------------------------

 ./
     README_SAD.txt           This file

 conf/
     mfcc_hires.conf          MFCC configuration

 exp/segmentation_1a/tdnn_stats_asr_sad_1a/
     final.raw                The pretrained model
     configs                  The neural network configs used
     post_output.vec          The labels priors vector
     cmvn_opts                CMVN opts used
     srand                    The RNG seed used


 II. Usage
 ------------------------------------------------------------------------------

 The neural network model in exp/segmentation_1a/tdnn_stats_asr_sad_1a/
 is to be used for speech activity detection using the script
 steps/segmentation/detect_speech_activity.sh.

 The usage command is 
 steps/segmentation/detect_speech_activity.sh [options] \
   --extra-left-context 79 --extra-right-context 21 \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --frames-per-chunk 150 --mfcc-config conf/mfcc_hires.conf \
   <src-data-dir> exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   <mfcc-dir> <work-dir> <out-data-dir>

 See steps/segmentation/detect_speech_activity.sh for more details about
 the options and usage.

 The network was trained using data from the following sources:
     Corpus                               LDC Catalog No.
     Fisher English Part 1 Speech         LDC2004S13
     Fisher English Part 2 Speech         LDC2005S13
     Fisher English Part 1 Transcripts    LDC2004T19
     Fisher English Part 2 Transcripts    LDC2005T19

 The following datasets were used in data augmentation.

     RIR_NOISES          http://www.openslr.org/28
