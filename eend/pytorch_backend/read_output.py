import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def write_output(filename, content):
    f = open("/home/sachini/HA-EEND/egs/librispeech/v1/exp/diarize/infer/real/output/"+filename, "w")
    f.write(" ".join(map(str, content)))
    f.close()

def read_output(filename):
    with h5py.File(filename, "r") as f:
        keys = list(f.keys())
        a_group_key = list(f.keys())[0]

        data = list(f[a_group_key])

    speaker1 = []
    speaker2 = []
    for each in data:
        speaker1.append(round(each[0]))
        speaker2.append(round(each[1]))
    
    write_output("speaker1", speaker1)
    write_output("speaker2", speaker2)

read_output(sys.argv[1])