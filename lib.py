
import requests
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd

load_dotenv()



class DataGetter():

  def __init__(self):
    self.BASE_URL = 'http://api.zoopla.co.uk/api/v1/property_listings.json?'
    self.API_KEY = os.getenv("ZOOPLA_API")

    if len(self.API_KEY) == 0:
      raise ValueError('Missing API key!')

  def buildURL(self, postcode, page=1):
    fixed_options = 'radius=0&category=residential&listing_status=rent&include_rented=1&page_size=100'
    url_option = "area=%s&%s&page_number=%s&api_key=%s" %(postcode, fixed_options, page, self.API_KEY)
    url = self.BASE_URL + url_option

    return url



  def getListing(self, url):

    response = requests.get(url)
    listing = []
    success = 0

    if response.status_code == 200:
      result = response.json()

      for i in range(0,len(result['listing'])):
        listing.append(result['listing'][i])

      success = 1
      print('Success')
    else:
      print('Error')
      print(url)
    return listing, success



  def getAllListings(self, postcode, from_page = 0, max_page = 1, find_pages = True):
    url = self.buildURL(postcode)
    response = requests.get(url)
    comb_listing = []

    if find_pages == True:
      result = response.json()
      max_page = int(round(result['result_count']/100,0))
    else:
      print("running API page %s to %s " %(from_page, max_page))


    for i in range(from_page, max_page):
      url = self.buildURL(postcode,i+1)
      new_listing, success = self.getListing(url)
      if success == 1:
        comb_listing += new_listing

    return comb_listing







