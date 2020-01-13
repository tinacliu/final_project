import lib
import pandas as pd
import numpy as np


def getBathBristolData():
# note API call is limited to 100 calls per hour
  postcodes = ['BA1','BA2','BS1','BS2','BS3','BS4','BS5','BS6','BS7','BS8','BS9']
  all_results = []
  call = lib.DataGetter()

  for postcode in postcodes:
      result = call.getAllListings(postcode)
      all_results += result
      print(postcode+' added')

  df = json_normalize(all_results)
  df.to_csv('./data/bath_bristol_rental.csv', index=False)

  print('data saved to csv in data folder')

  return df



def getLondonData():
  all_results = []
  call = lib.DataGetter()

  result = call.getAllListings('London',0,99, False)
  all_results += result

  result = call.getAllListings('NW',0,99, False)
  all_results += result

  result = call.getAllListings('SE',0,48, False)
  all_results += result

  result = call.getAllListings('SW',0,99, False)
  all_results += result

  postcodes = []
  for i in range(1,23):
      postcodes.append("E%s" %i)
      postcodes.append("N%s" %i)

  for i in range(1,15):
      postcodes.append("E%s" %i)

  for postcode in postcodes:
    result = call.getAllListings(postcode)
    all_results += result
    print(postcode +' added')

  df = json_normalize(all_results)
  df.to_csv('./data/london_rental.csv', index=False)

  print('data saved to csv in data folder')

  return df



def cleanBBData():
  """
  Load the data from CSV and conduct cleaning, steps including:
  <p>1.1 only load in the data columns where information is useful i.e. getting rid of irrelavant data saved from Zoopla API call</p>
  <p>1.2 check for missing values in df</p>
  <p>1.3 For each data series, convert Nulls, check for validity of data including checking for extreme values / errors</p>
  <p>1.4 delete duplicates</p>
  """

  # triming the original 56 columns to 17 useful columns
  cols = ['bills_included','description',
        'details_url','first_published_date',
        'floor_plan', 'num_bathrooms','num_bedrooms','num_recepts',
        'furnished_state', 'property_type', 'rental_prices.shared_occupancy',
        'latitude', 'longitude','outcode',
        'listing_id','status',
        'rental_prices.per_month']

  df = pd.read_csv('./data/bath_bristol_rental.csv', usecols=cols)


  # rename the columns
  df.columns = ['bills_included', 'description', 'details_url', 'first_published_date',
              'floor_plan', 'furnished_state', 'latitude', 'listing_id', 'longitude',
              'num_bathrooms', 'num_bedrooms', 'num_recepts', 'outcode',
              'property_type', 'rent_price', 'shared_occu','status'
             ]

  # data series with null columns are: bills_included, description, floor_plan, furnished_state, property_type

  # clean Nulls in bills_included
  df['bills_included'].fillna(value=0, inplace=True)

  # clean Nulls in description
  df['description'].fillna(value='No description', inplace=True)

  # clean Nulls and make floor_plan a binary column to state if floorplan is available
  df['floor_plan'].fillna(value=0, inplace=True)
  df['floor_plan'] = df['floor_plan'].map(lambda x: 1 if x != 0 else 0)

  # generate a new column indicating student property
  df['student'] = df['description'].map(lambda x: 1 if 'student' in x else 0)

  # if furnished state is missing and its a student property, assume its furnithsed
  df['furnished_state']  = np.where((df['furnished_state'].isna() & df['student'] == 1),
                                    'furnished', df['furnished_state'])

  df['furnished_state'].fillna(value='Missing', inplace=True)

  # turn shared occupency to binary
  df['shared_occu'] = df['shared_occu'].map(lambda x: 1 if x == 'Y' else 0)

  # after investigation, the 11 bathroom one is an error
  df.drop(1835, inplace=True)


  # clean up and combine property categories together that means the same thing
  def type_classify(x):

      flats = ['Flat','Maisonette']
      terrace = ['Terraced house','End terrace house','Town house','Mews house']
      country = ['Cottage','Detached bungalow','Barn conversion','Semi-detached bungalow']

      if x in flats:
          return 'Flats'
      elif x in terrace:
          return 'Terrace'
      elif x in country:
          return 'Country'
      elif x == np.nan:
          return 'NA'
      else:
          return x

  df['property_type'] = df['property_type'].map(lambda x: type_classify(x))

  # delete all rows with property type "parking/garage"
  indexNames = df[df['property_type'] == 'Parking/garage'].index
  df.drop(indexNames, inplace=True)

  df['property_type'].fillna(value='Missing', inplace=True)

  df['rented'] = df['status'].map(lambda x: 0 if x in ['to_rent'] else 1)
  df.drop(['status'], axis=1, inplace=True)

  df.drop_duplicates(subset='listing_id', keep='first', inplace=True)

  return df



def per_room_price(x):
  if (x['num_bedrooms'] == 0) & (x['property_type'] == 'Studio'):
      return x['rent_price']
  elif x['num_bedrooms'] == 0:
      return 0
  else:
      return round(x['rent_price']/(x['num_bedrooms']),0)



def getStudentRentals(df):

  student = df[(df['student']==1)]

  student['room_pm'] = student.apply(lambda x: per_room_price(x), axis=1)

  return student



def getPerBedroomOvreview(region):
  grouped = region.groupby(['num_bedrooms']) \
         .agg({'listing_id':'size', 'bills_included':'sum','floor_plan':'sum','room_pm':'median','num_bathrooms':'mean'}) \
         .reset_index() \
         .rename(columns={'listing_id':'count','num_bathrooms':'avg_baths'})

  total = grouped['count'].sum()

  grouped['proportion%'] = grouped['count'].map(lambda x: round(x/total*100,1) )

  return grouped



def getBathroomsByProperSize(region):

  grouped = region.groupby(['num_bedrooms','num_bathrooms']) \
         .agg({'listing_id':'size','room_pm':'median'}) \
         .rename(columns={'listing_id':'count'})

  grouped_perc = grouped.groupby(level=0)[['count']].apply(lambda x: round(100 * x / x.sum(),1))

  joined = grouped_perc.reset_index().join(grouped.reset_index(), lsuffix='_l', rsuffix='_r')

  bed_bath = joined[['num_bedrooms_l','num_bathrooms_l','count_r','count_l','room_pm']]
  bed_bath.columns =['num_bedrooms','num_bathrooms','count','proportion%','rent_pm']

  return bed_bath
