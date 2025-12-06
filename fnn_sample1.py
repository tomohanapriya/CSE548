"""
Created on Mon Apr 15 19:43:04 2019
Updated on Wed Jan 29 10:18:09 2020
@author: created by Sowmya Myneni and updated by Dijiang Huang
"""
"""

########################################
# Part 1 - Datasets Pre-Processing
#######################################
# To load a dataset file in Python, you can use Pandas. Import
pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np
# Controls which scenario we want to run.
# Accepts a, b or c as input
ScenarioA = ['Training-a1-a3', 'Testing-a2-a4']
ScenarioB = ['Training-a1-a2', 'Testing-a1']
ScenarioC = ['Training-a1-a2', 'Testing-a1-a2-a3']
while 1:
Scenario = input ('Please enter the scenario you wish to run -
either a, b or c:')
if Scenario.lower() == 'a':
TrainingData = ScenarioA[0]
TestingData = ScenarioA[1]
break
elif Scenario.lower() == 'b':
TrainingData = ScenarioB[0]
TestingData = ScenarioB[1]
break
elif Scenario.lower() == 'c':
TrainingData = ScenarioC[0]
TestingData = ScenarioC[1]
break
# Batch Size
This study source was downloaded by 100000869502194 from CourseHero.com on 11-30-2025 19:23:14 GMT -06:00

https://www.coursehero.com/file/210839344/fnn-samplepy/
