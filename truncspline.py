from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import pandas as pd

wagedf = pd.read_csv(index_col = 0, filepath_or_buffer = "https://raw.githubusercontent.com/puria-radmard/ISLR-python/master/Notebooks/Data/Wage.csv")


class TruncatedSpline:
    
    def __init__(self, degree = 3):
        """
       
        
        Degree is the power to which the table is produced.
        """
        
        self.degree = degree
    
    
    def fit(self, X, y, knots):
        
        """
        knots comes in the form {"Parameter1": [k1, k2, k3], "Parameter2": [k4, k5, k6]}
        where Parameters are the model predictors, and k# are the values at which the knots
        exist for their respective predictor.
        
        If no knots are used but the parameter is still to be used in the regression,
        simply pass {... , Parameterj:[] ,  ...}
        """
        
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(data = X)
        
        self.cols = list(knots.keys())
        
        self.truncs = {}
        for param in knots:
            
            for k in knots[param]:
                
                self.truncs["{}_{}".format(param, k)] = """(df['{}'] - {}) * (df['{}'].apply(lambda x: int(x > {})))""".format(
                    param, k, param, k)
                
        return self
                
                
    def transform(self, X):
        
        df = X.copy()[self.cols]
        
        for trunc in self.truncs:
            for n in range(1, self.degree + 1):
                
                df["{}_^{}".format(trunc, n)] = eval(self.truncs[trunc])**n
                          
        for col in self.cols:
            for n in range(2, self.degree + 1):
                
                df["{}_^{}".format(col, n)] = df[col]**n
        
        return (pd.concat((X, df), axis = 1))

#%%
testdf = pd.get_dummies(wagedf[["age", "race", "wage"]]).copy()
testdf.head()
a = TruncatedSpline(degree = 3).fit(X = pd.get_dummies(wagedf[["age", "race"]]), y = testdf["wage"],  knots = {"age": [40, 60]})
a.transform(testdf)


#%%
pipe = make_pipeline(TruncatedSpline(), LinearRegression())
pipe.fit(pd.get_dummies(wagedf[["age", "race"]]), wagedf["wage"], truncatedspline__knots = {"age": [30, 40, 50, 60, 70]})
pipe.named_steps['truncatedspline']

#%%
### Without Splining

predicted_wages = LinearRegression().fit(pd.get_dummies(wagedf[["age", "race"]]), 
                                         wagedf["wage"]).predict(pd.get_dummies(wagedf[["age", "race"]]))

sns.scatterplot(wagedf["age"], wagedf["wage"], s= 10)
sns.lineplot(wagedf["age"], predicted_wages)

#%%
### With Splining

predicted_wages = pipe.predict(pd.get_dummies(wagedf[["age", "race"]]))

sns.scatterplot(wagedf["age"], wagedf["wage"], s= 10)
sns.lineplot(wagedf["age"], predicted_wages)