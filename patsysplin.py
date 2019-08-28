import numpy as numpy
import pandas as pd
from patsy import dmatrix

class SplineFitter:
    
    def __init__(self, method, param, degree = None, include_intercept = False):
        """
        method is knot, df, or natural
        """
        self.method = method
        self.param = param
        self.ii = include_intercept
        self.degree = degree
    
    def fit(self, X, y, column):
        self.column = column
        return self
    
    def transform(self, X):
        
        if type(X) == pd.DataFrame and len(X.columns)>1:
            save = X.drop(self.column, axis = 1)
        
        X_ = X.copy()
        
        if self.method == "knot":
            f = "bs(X_.{}, knots={}, degree={}, include_intercept={})".format(self.column, self.param, self.degree, self.ii)
        
        elif self.method == "df":
            f = "bs(X_.{}, df={}, degree={}, include_intercept={})".format(self.column, self.param, self.degree, self.ii)
        
        elif self.method == "natural":
            f = "cr(X_.{}, df={})".format(self.column, self.param)
            
        splined_ = dmatrix(f, 
                           {"X_.{}".format(self.column): X_[self.column]},
                           return_type = "dataframe")
        
        if type(X) == pd.DataFrame and len(X.columns)>1:  
            outdf = pd.concat((save, splined_), axis = 1)
        else:
            outdf = splined_
            
        return outdf