import numpy as np
import joblib as jbl
import os
from flask import Flask, request, render_template, make_response

# Runing application
app = Flask( __name__, static_url_path = '', template_folder = '' )

# Load model
model = jbl.load( 'model.pkl' )

@app.route( '/' )
def display_gui():
  return render_template( 'form.html' )

@app.route( '/get_price', methods=['POST'] )

def get_price():
  area = request.form['area']
  rooms = request.form['rooms']
  bathroom = request.form['bathroom']
  parking_spaces = request.form['parking_spaces']
  floor = request.form['floor']
  accept_animal_yes = 1 if request.form['accept_animal'] == 1 else 0
  accept_animal_no = 0 if request.form['accept_animal'] == 0 else 1
  furnished_yes = 1 if request.form['furnished'] == 1 else 0
  furnished_no = 0 if request.form['furnished'] == 0 else 1
  rent_amount = request.form['rent_amount']
  property_tax = request.form['property_tax']
  
  params = np.array( [ [ area, rooms, bathroom, parking_spaces,floor, accept_animal_yes, accept_animal_no, furnished_yes, furnished_no, rent_amount, property_tax ] ] )
  print( params )

  price = model.predict( params )[0]
  print( "Estimated price: {}".format( str( price ) ) )

  return render_template( 'form.html', price = str( price ) )

if __name__ == "__main__":
  port = int( os.environ.get( 'PORT', 5500 ) )
  app.run( host='0.0.0.0', port=port )