class TruncatedSpline:
    
    def __init__(self, degree = 3):
        """
       
        
        Degree is the power to which the table is produced.
        """
        
        self.degree = degree
    
    
    def fit(self, X, knots):
        
        """
        knots comes in the form {"Parameter1": [k1, k2, k3], "Parameter2": [k4, k5, k6]}
        where Parameters are the model predictors, and k# are the values at which the knots
        exist for their respective predictor.
        """
        
        self.cols = list(knots.keys())
        
        self.truncs = {}
        for param in knots:
            
            for k in knots[param]:
                
                self.truncs["{}_{}".format(param, k)] = """(df['{}'] - {}) * (df['{}'].apply(lambda x: int(x > {})))""".format(
                    param, k, param, k)
                
        return self
                
                
    def transform(self, X):
        
        df = X
        
        for trunc in self.truncs:
            for n in range(1, self.degree + 1):
                
                df["{}_^{}".format(trunc, n)] = eval(self.truncs[trunc])**n
                          
        for col in self.cols:
            for n in range(2, self.degree + 1):
                
                df["{}_^{}".format(col, n)] = df[col]**n
        
        return (pd.concat((X, df), axis = 1))