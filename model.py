import pathlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib as jbl

# load dataset
file_path = str( pathlib.Path().absolute() ) + '/dataset/'
file_name = 'PP-2020-05-21-22-24-29-houses_to_rent_v2.csv'

df = pd.read_csv( file_path + file_name )

# get predicts var
x = df.drop('total', axis = 1)
# get target var
y = df['total']

# instance model
#model =  LinearRegression()
#model = RandomForestRegressor( n_estimators = 200, n_jobs = -1 )
model = GradientBoostingRegressor( n_estimators = 200 )

# method train test validation
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 2 )
model.fit( x_train, y_train )

result = model.score( x_test, y_test )

print( result )

jbl.dump( model, 'model.pkl' )
