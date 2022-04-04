import random
import pandas as pd
import numpy as np
from MarkovChain import *
from MChainDraw import *
from statistics import mean
from probability import *

"""
Name: Luciano Zavala
Date: 04/03/22
Assignment: Module 11: Project - MM and HMM
Due Date: 04/03/22
About this project: python script that computes a Hidden Markov Model
Assumptions:NA
All work below was performed by LZZ
"""

random.seed("aima-python")
df = pd.read_excel("PRSA_Data_Gucheng_20130301-20170228.xlsx")
print("       General Information")
print("-------------------------------------")
print(df.info())

print("\n          CO Statistics")
print("----------------------------------------")
print(df['CO'].describe())


def label_CO(row):
    if row['CO'] <= 600:
        return "low"
    elif 600 < row['CO'] <= 900:
        return "medium"
    elif 900 < row['CO'] <= 1600:
        return "high"
    elif row['CO'] > 1600:
        return "extreme"
    return "Other"


df['CO_label'] = df.apply(lambda row: label_CO(row), axis=1)
print(df['CO_label'])

print("\n          Labels")
print("----------------------------------------")
print(df['CO_label'].unique())
print(df['CO_label'].nunique())
print(df['CO_label'].describe())

CountLowLow = 0  # ok
CountLowMedium = 0
CountLowHigh = 0
CountLowExtreme = 0

CountMediumMedium = 0
CountMediumLow = 0
CountMediumHigh = 0
CountMediumExtreme = 0

CountHighHigh = 0
CountHighLow = 0
CountHighMedium = 0
CountHighExtreme = 0

CountExtremeExtreme = 0
CountExtremeLow = 0
CountExtremeMedium = 0
CountExtremeHigh = 0

indexCOLabel = df.columns.get_loc("CO_label")

for i in range(1, df.shape[0] - 1):
    if df.iat[i, indexCOLabel] == 'low':
        if df.iat[i + 1, indexCOLabel] == 'low':
            CountLowLow += 1
        elif df.iat[i + 1, indexCOLabel] == 'medium':
            CountLowMedium += 1
        elif df.iat[i + 1, indexCOLabel] == 'high':
            CountLowHigh += 1
        elif df.iat[i + 1, indexCOLabel] == 'extreme':
            CountLowExtreme += 1
    elif df.iat[i, indexCOLabel] == 'medium':
        if df.iat[i + 1, indexCOLabel] == 'low':
            CountMediumLow += 1
        elif df.iat[i + 1, indexCOLabel] == 'medium':
            CountMediumMedium += 1
        elif df.iat[i + 1, indexCOLabel] == 'high':
            CountMediumHigh += 1
        elif df.iat[i + 1, indexCOLabel] == 'extreme':
            CountMediumExtreme += 1
    elif df.iat[i, indexCOLabel] == 'high':
        if df.iat[i + 1, indexCOLabel] == 'low':
            CountHighLow += 1
        elif df.iat[i + 1, indexCOLabel] == 'medium':
            CountHighMedium += 1
        elif df.iat[i + 1, indexCOLabel] == 'high':
            CountHighHigh += 1
        elif df.iat[i + 1, indexCOLabel] == 'extreme':
            CountHighExtreme += 1
    elif df.iat[i, indexCOLabel] == 'extreme':
        if df.iat[i + 1, indexCOLabel] == 'low':
            CountExtremeLow += 1
        elif df.iat[i + 1, indexCOLabel] == 'medium':
            CountExtremeMedium += 1
        elif df.iat[i + 1, indexCOLabel] == 'high':
            CountExtremeHigh += 1
        elif df.iat[i + 1, indexCOLabel] == 'extreme':
            CountExtremeExtreme += 1

print("\nCase Count:")
print("---------------")

print("\nlow:")
print(CountLowLow)
print(CountLowMedium)
print(CountLowHigh)
print(CountLowExtreme)
print("\nmedium:")
print(CountMediumMedium)
print(CountMediumLow)
print(CountMediumHigh)
print(CountMediumExtreme)
print("\nhigh:")
print(CountHighHigh)
print(CountHighLow)
print(CountHighMedium)
print(CountHighExtreme)
print("\nextreme:")
print(CountExtremeExtreme)
print(CountExtremeLow)
print(CountExtremeMedium)
print(CountExtremeHigh)

##################################
#   Probabilities computation    #
##################################

ProbLowLow = CountLowLow / (CountLowLow + CountLowMedium + CountLowHigh + CountLowExtreme)
ProbLowMedium = CountLowMedium / (CountLowLow + CountLowMedium + CountLowHigh + CountLowExtreme)
ProbLowHigh = CountLowHigh / (CountLowLow + CountLowMedium + CountLowHigh + CountLowExtreme)
ProbLowExtreme = CountLowExtreme / (CountLowLow + CountLowMedium + CountLowHigh + CountLowExtreme)

# medium

ProbMediumMedium = CountMediumMedium / (CountMediumMedium +
                                        CountMediumLow +
                                        CountMediumHigh +
                                        CountMediumExtreme)

ProbMediumLow = CountMediumLow / (CountMediumMedium +
                                  CountMediumLow +
                                  CountMediumHigh +
                                  CountMediumExtreme)

ProbMediumHigh = CountMediumHigh / (CountMediumMedium +
                                    CountMediumLow +
                                    CountMediumHigh +
                                    CountMediumExtreme)

ProbMediumExtreme = CountMediumExtreme / (CountMediumMedium +
                                          CountMediumLow +
                                          CountMediumHigh +
                                          CountMediumExtreme)
# High

ProbHighHigh = CountHighHigh / (CountHighHigh +
                                CountHighLow +
                                CountHighMedium +
                                CountHighExtreme)

ProbHighLow = CountHighLow / (CountHighHigh +
                              CountHighLow +
                              CountHighMedium +
                              CountHighExtreme)

ProbHighMedium = CountHighMedium / (CountHighHigh +
                                    CountHighLow +
                                    CountHighMedium +
                                    CountHighExtreme)

ProbHighExtreme = CountHighExtreme / (CountHighHigh +
                                      CountHighLow +
                                      CountHighMedium +
                                      CountHighExtreme)

# Extreme

ProbExtremeExtreme = CountExtremeExtreme / (CountExtremeExtreme +
                                            CountExtremeLow +
                                            CountExtremeMedium +
                                            CountExtremeHigh)

ProbExtremeLow = CountExtremeLow / (CountExtremeExtreme +
                                    CountExtremeLow +
                                    CountExtremeMedium +
                                    CountExtremeHigh)

ProbExtremeMedium = CountExtremeMedium / (CountExtremeExtreme +
                                          CountExtremeLow +
                                          CountExtremeMedium +
                                          CountExtremeHigh)

ProbExtremeHigh = CountExtremeHigh / (CountExtremeExtreme +
                                      CountExtremeLow +
                                      CountExtremeMedium +
                                      CountExtremeHigh)


# Compute sensors
def label_CO_evidence(row):
    if row['month'] in range(9, 11):
        return True
    return False


df['CO_month_evidence'] = df.apply(lambda row: label_CO_evidence(row), axis=1)

CountTTTT = 0
CountTTTF = 0
CountTTFT = 0
CountTTFF = 0

CountTFTT = 0
CountTFTF = 0
CountTFFT = 0
CountTFFF = 0

CountFTTT = 0
CountFTTF = 0
CountFTFT = 0
CountFTFF = 0

CountFFTT = 0
CountFFTF = 0
CountFFFT = 0
CountFFFF = 0

indexCOEvidenceLabel = df.columns.get_loc("CO_month_evidence")

'''
for i in range(1, df.shape[0] - 1):
    if df.iat[i, indexCOEvidenceLabel]:
        if df.iat[i + 1, indexCOEvidenceLabel]:
            if df.iat[i + 2, indexCOEvidenceLabel]:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountTTTT += 1
                else:
                    CountTTTF += 1

            else:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountTTFT += 1
                else:
                    CountTTFF += 1

        else:
            if df.iat[i + 2, indexCOEvidenceLabel]:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountTFTT += 1
                else:
                    CountTFTF += 1
            else:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountTFFT += 1
                else:
                    CountTFFF += 1

    else:
        if df.iat[i + 1, indexCOEvidenceLabel]:
            if df.iat[i + 2, indexCOEvidenceLabel]:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountFTTT += 1
                else:
                    CountFTTF += 1
            else:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountFTFT += 1
                else:
                    CountFTFF = 0
        else:
            if df.iat[i + 2, indexCOEvidenceLabel]:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountFFTT += 1
                else:
                    CountFFTF += 1
            else:
                if df.iat[i + 3, indexCOEvidenceLabel]:
                    CountFFFT = 0
                else:
                    CountFFFF = 0


ProbTTTT = CountTTTT / (CountTTTT +
                        CountTTTF +
                        CountTTFT +
                        CountTTFF)


ProbTTTF = CountTTTF / (CountTTTT +
                        CountTTTF +
                        CountTTFT +
                        CountTTFF)

ProbTTFT = CountTTFT / (CountTTTT +
                        CountTTTF +
                        CountTTFT +
                        CountTTFF)


ProbTTFF = CountTTFF / (CountTTTT +
                        CountTTTF +
                        CountTTFT +
                        CountTTFF)


ProbTFTT = CountTFTT / (CountTFTT +
                        CountTFTF +
                        CountTFFT +
                        CountTFFF)


ProbTFTF = CountTFTF / (CountTFTT +
                        CountTFTF +
                        CountTFFT +
                        CountTFFF)


ProbTFFT = CountTFFT / (CountTFTT +
                        CountTFTF +
                        CountTFFT +
                        CountTFFF)


ProbTFFF = CountTFFF / (CountTFTT +
                        CountTFTF +
                        CountTFFT +
                        CountTFFF)


ProbFTTT = CountFTTT / (CountFTTT +
                        CountFTTF +
                        CountFTFT +
                        CountFTFF)


ProbFTTF = CountFTTF / (CountFTTT +
                        CountFTTF +
                        CountFTFT +
                        CountFTFF)


ProbFTFT = CountFTFT / (CountFTTT +
                        CountFTTF +
                        CountFTFT +
                        CountFTFF)

ProbFTFF = CountFTFF / (CountFTTT +
                        CountFTTF +
                        CountFTFT +
                        CountFTFF)

ProbFFTT = CountFFTT / (CountFFTT +
                        CountFFTF +
                        CountFFFT +
                        CountFFFF)


ProbFFTF = CountFFTF / (CountFFTT +
                        CountFFTF +
                        CountFFFT +
                        CountFFFF)

ProbFFFT = CountFFFT / (CountFFTT +
                        CountFFTF +
                        CountFFFT +
                        CountFFFF)

ProbFFFF = CountFFFF / (CountFFTT +
                        CountFFTF +
                        CountFFFT +
                        CountFFFF)
'''

CountTT = 0
CountTF = 0
CountFF = 0
CountFT = 0

for i in range(1, df.shape[0] - 1):
    if df.iat[i, indexCOEvidenceLabel]:
        if df.iat[i + 1, indexCOEvidenceLabel]:
            CountTT += 1
        else:
            CountTF += 1
    else:
        if df.iat[i + 1, indexCOEvidenceLabel]:
            CountFT += 1
        else:
            CountFF += 1

ProbTT = CountTT / (CountTT + CountTF)
ProbTF = CountTF / (CountTT + CountTF)
ProbFT = CountFT / (CountFT + CountFF)
ProbFF = CountFF / (CountFT + CountFF)

transition = [
    [ProbLowLow, ProbLowMedium, ProbLowHigh, ProbLowExtreme],
    [ProbMediumLow, ProbMediumMedium, ProbMediumHigh, ProbMediumExtreme],
    [ProbHighLow, ProbHighMedium, ProbHighHigh, ProbHighExtreme],
    [ProbExtremeLow, ProbExtremeMedium, ProbExtremeHigh, ProbExtremeExtreme]
]
sensor = [[ProbTT, ProbFF],
          [ProbTF, ProbFT]]

CORateHMM = HiddenMarkovModel(transition, sensor)
print(CORateHMM.__dict__)
evidence = [T, T, T, T, F]

print(rounder(forwardOnly(CORateHMM, evidence)))
