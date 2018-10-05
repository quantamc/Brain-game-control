import numpy as numpy
import pandas as pd
import glob

path = "./ag"
filenames = glob.glob(path+ "/*.csv")
frame = pd.DataFrame()
for file_ in filenames:
    df = 


def sample(sequence, label):
    tot = sequence.shape[0]
    i = 0

    for range(tot):
        sequence


def load_data(frame):

    # drop the duplicates
    frame = frame.drop_duplicates()
    instances = frame.shape[0]
    # grab the sensor values
    