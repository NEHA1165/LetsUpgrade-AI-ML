import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

dataset1=pd.read_excel("my_data.xlsx",sheet_name=0)
print(dataset1.head())
print(dataset1.columns)

print(dataset1.isnull())

print(dataset1.duplicated())

print(dataset1.drop_duplicates())

dataset2=dataset1[['Age','DistanceFromHome','Education','MonthlyIncome',
'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear',
'YearsAtCompany','YearsSinceLastPromotion', 'YearsWithCurrManager']].describe()

print(dataset2)

stats,p=pearsonr(dataset1.Attrition,dataset1.Age)
print(stats,p)

stats,p1=pearsonr(dataset1.Attrition,dataset1.DistanceFromHome)
print(stats,p1)

stats,p2=pearsonr(dataset1.Attrition,dataset1.Education)
print(stats,p2)

stats,p3=pearsonr(dataset1.Attrition,dataset1.JobLevel)
print(stats,p3)

stats,p4=pearsonr(dataset1.Attrition,dataset1.MonthlyIncome)
print(stats,p4)

stats,p5=pearsonr(dataset1.Attrition,dataset1.StockOptionLevel)
print(stats,p5)

stats,p6=pearsonr(dataset1.Attrition,dataset1.YearsWithCurrManager)
print(stats,p6)

stats,p7=pearsonr(dataset1.Attrition,dataset1.YearsAtCompany)
print(stats,p7)

stats,p8=pearsonr(dataset1.Attrition,dataset1.PercentSalaryHike)
print(stats,p8)

stats,p9=pearsonr(dataset1.Attrition,dataset1.NumCompaniesWorked)
print(stats,p9)
