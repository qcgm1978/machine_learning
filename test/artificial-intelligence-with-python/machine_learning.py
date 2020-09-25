# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
cols='''Index Date         Location   MinTemp ... RainToday  RISK_MM  RainTomorrow'''
s='''0   2008-12-01   Albury           13.4 ...         No             0.0                   No
1   2008-12-02   Albury            7.4 ...          No             0.0                   No
2   2008-12-03   Albury          12.9 ...          No             0.0                   No
3   2008-12-04   Albury            9.2 ...          No             1.0                   No
4   2008-12-05   Albury          17.5 ...          No             0.2                   No'''
l=re.split(r'\n',s)
rows=list(map(lambda item:re.split(r'\s+',item),l))
columns=re.split(r'\s+',cols)
# print(rows,columns)
df = pd.DataFrame(np.array(rows),
    columns=columns
)
ret=df.to_csv('/Users/zhanghongliang/Documents/machine_learning/test/artificial-intelligence-with-python/data/weatherAUS.csv',index=False)
#Load the data set
df = pd.read_csv('/Users/zhanghongliang/Documents/machine_learning/test/artificial-intelligence-with-python/data/weatherAUS.csv')

 