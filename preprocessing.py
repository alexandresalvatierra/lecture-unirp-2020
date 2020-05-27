import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from tabulate import tabulate as tb
from datetime import datetime

file_path = str( Path(__file__).parent.absolute() ) + '/dataset/'
file_name = 'houses_to_rent_v2.csv'

# read external csv
pd.set_option( 'display.max_columns', 13 )
df = pd.read_csv( file_path + file_name )

df.drop( 'city', axis = 1, inplace = True )

# rename columns
df.rename({'parking spaces': 'parking_spaces'}, axis=1, inplace=True)
df.rename({'hoa (R$)': 'hoa', 'rent amount (R$)': 'rent_amount'}, axis=1, inplace=True)
df.rename({'property tax (R$)': 'property_tax', 'fire insurance (R$)': 'fire_insu'}, axis=1, inplace=True)
df.rename({'total (R$)': 'total'}, axis=1, inplace=True)

print( tb( df.head(), headers = 'keys', tablefmt = 'grid' ) )

# show types of columns
print( df.dtypes )

# change type of column as int
df['floor'] = df['floor'].str.replace( '-','1' )
df['floor'] = df['floor'].astype( int )

# show types of columns
#print( df.dtypes )

# function to show missing data
def getMissingInfo( df ):
    # get amount missing data
    missing = df.isnull().sum()
    print( missing )
    # get percent missing data
    missing_percent = ( missing / len( df[df.columns[0]] ) ) * 100
    print( missing_percent )

getMissingInfo( df )

# function to calc Outliers IRQ method
def getIRQ( column ):
    # arrange data in increasing order
    sort_column = column.sort_values(ascending=True)

    # find first quartile and third quartile
    q1, q3 = sort_column.quantile( [ 0.25, 0.75 ] )

    # find IQR - difference between third and first quartile
    iqr = q3 - q1

    # find lower and upper bound
    lower_bound = q1 - ( 1.5 * iqr ) 
    upper_bound = q3 + ( 1.5 * iqr ) 

    return lower_bound, upper_bound

lb, ub = getIRQ( df['total'] )

# Show lower and upper bounds
print("Lower bound: ", lb, " Upper bound: ", ub)

# Show total of Uppers Outliers
print( "Total Uppers Outliers: ", df[  df['total'] > ub  ].count()['total'] )

# Show total of Lowers Outliers
print( "Total Lowers Outliers: ", df[  df['total'] < lb  ].count()['total'] )

df.boxplot( column = ['total'] )
plt.show()

# function to show correlation
def getCorrelation( df ):
  # show correlation
  print( df.corr( method = 'pearson' ) )
  # set plot setup
  plt.figure( figsize = (7, 7) )
  # set heatmap setup
  sb.heatmap( df.corr( method = 'pearson' ) )
  # show image
  plt.show()

getCorrelation( df )

# remove correlated coluns
df.drop( ['hoa','fire_insu'], axis = 1, inplace = True )

getCorrelation( df )

# Set Category Vars on numbers (one hot encode)
animal = pd.get_dummies( df['animal'] )
furniture = pd.get_dummies( df['furniture'] )

df.drop( ['animal', 'furniture'], axis = 1, inplace = True )

# Set one hot encode vars on dataset
df_concat = pd.concat( [df, animal, furniture], axis = 1 )

df = df_concat

print( tb( df.head(), headers = 'keys', tablefmt = 'grid' ) )

now = datetime.now()
dt_str = now.strftime("%Y-%m-%d-%H-%M-%S")
df_concat.to_csv(file_path + 'PP-' + dt_str + '-' + file_name, index = False)