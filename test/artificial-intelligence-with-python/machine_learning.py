# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
class Panda(object):
    def DataFrame(self,s,cols):
        l=re.split(r'\n',s)
        rows=list(map(lambda item:re.split(r'\s+',item),l))
        columns=re.split(r'\s+',cols)
        # print(rows,columns)
        df = pd.DataFrame(np.array(rows),
            columns=columns     
        )
        return df
    def to_csv(self,df,p):
        df.to_csv(p,index=False)
    def read_csv(self):
        #Load the data set
        df = pd.read_csv('/Users/zhanghongliang/Documents/machine_learning/test/artificial-intelligence-with-python/data/weatherAUS.csv')

 