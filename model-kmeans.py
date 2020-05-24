from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd

# load dataset
df = load_breast_cancer()
# predict vars
x = pd.DataFrame( df.data, columns = [df.feature_names] )
# target vars
y = pd.Series( df.target )

# normalization
norm = MinMaxScaler( feature_range = (0, 1) )
x_norm = norm.fit_transform( x )

# instance model
model = KMeans( n_clusters = 2, random_state = 16 )
# train model
model.fit( x_norm )

# show centroids
print( model.cluster_centers_ )

clusters = model.predict( x_norm )
# show predict
print( clusters )

# compare arrays
res = accuracy_score( y, clusters )

# show result
print( res )