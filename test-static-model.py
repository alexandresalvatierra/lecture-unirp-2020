import numpy as np
import joblib as jbl

model = jbl.load('model.pkl')

data_test = np.array( [[80,1,1,1,6,2800,0,1,0,0,1]] )

print( model.predict( data_test ) )