"""
Name: Luciano Zavala
Date: 04/03/22
Assignment: Module 11: Project - MM and HMM
Due Date: 04/03/22
About this project: python script that computes a Markov Model and graphs it.
Assumptions:NA
All work below was performed by LZZ
"""

import random
import pandas as pd
import numpy as np
from MarkovChain import *
from MChainDraw import *
from statistics import mean

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

##################################
#         Markov Model           #
##################################

transition_prob = {
    'low': {'low': ProbLowLow,
            'medium': ProbLowMedium,
            'high': ProbLowHigh,
            'extreme': ProbLowExtreme
            },
    'medium': {'low': ProbMediumLow,
               'medium': ProbMediumMedium,
               'high': ProbMediumHigh,
               'extreme': ProbMediumExtreme
               },
    'high': {'low': ProbHighLow,
             'medium': ProbHighMedium,
             'high': ProbHighHigh,
             'extreme': ProbHighExtreme
             },
    'extreme': {'low': ProbExtremeLow,
                'medium': ProbExtremeMedium,
                'high': ProbExtremeHigh,
                'extreme': ProbExtremeExtreme
                }
}

CORate_chain = MarkovChain(transition_prob=transition_prob)

print("\n     MARKOV MODEL TESTS")
print("--------------------------------")
print(CORate_chain.__dict__)

print(CORate_chain.next_state(current_state='low'))
print(CORate_chain.next_state(current_state='high'))
print(CORate_chain.next_state(current_state='extreme'))
print(CORate_chain.generate_states(current_state='high', no=25))

# Model Array for graphic purpose
P = np.array([
    [round(ProbLowLow, 2), round(ProbLowMedium, 2), round(ProbLowHigh, 2), round(ProbLowExtreme, 2)],
    [round(ProbMediumLow, 2), round(ProbMediumMedium, 2), round(ProbMediumHigh, 2), round(ProbMediumExtreme, 2)],
    [round(ProbHighLow, 2), round(ProbHighMedium, 2), round(ProbHighHigh, 2), round(ProbHighExtreme, 2)],
    [round(ProbExtremeLow, 2), round(ProbExtremeMedium, 2), round(ProbExtremeHigh, 2), round(ProbExtremeExtreme, 2)]
])

mc = MarkovChainDraw(P, ['low', 'medium', 'high', 'extreme'])
mc.draw()
