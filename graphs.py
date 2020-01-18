import lib, data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')

def rentPriceGraphs(data,outcode, price, min_rent,max_rent):

  # sns.set(style="whitegrid")
  fig = plt.figure(figsize = (14,8))
  sns.set_context("poster", font_scale = 0.6, rc={"grid.linewidth": 1,'xtick.minor.width': 0.8})

  # plot1 rented versus non-rented boxplot
  ax1 = fig.add_subplot(121)
#     ax1.set_xticks(range(min_rent,max_rent,50), minor=True)
  sns.boxplot(x="rented", y=price, data=data, linewidth=1, width=0.5);
  ax1.set_ylabel('Per Room Rental Price')
  ax1.set_xlabel('Property Rented')
  ax1.set_ylim([min_rent, max_rent])
  plt.title("%s Rental - Rented VS. Available" %outcode)

  # plot2 rent versus bathroom
  ax2 = fig.add_subplot(122)
  sns.boxplot(x="num_bedrooms", y=price,
              data=data, linewidth=1, width=0.5);
  ax2.set_xlabel('Number of Bedrooms')
  ax2.set_ylabel('Per Room Rental Price')
  ax2.set_ylim([min_rent, max_rent])
  plt.title("%s Rental by Num Bedrooms" %outcode)

  # return '-'


def rentPriceBedBath(data, outcode, price, min_rent, max_rent):

  fig = plt.figure(figsize = (12,10))
  plt.subplots_adjust(top = 0.95, bottom=0.05, hspace=0.4, wspace=0.4)

  sns.set_context("poster", font_scale = 0.6, rc={"grid.linewidth": 1,'xtick.minor.width': 0.8})

  # plot1 rented versus non-rented boxplot
  for i in [1,2,3,4]:
    ax = fig.add_subplot("22%s" %i)
    sns.boxplot(x='num_bathrooms', y=price, data=data[data['num_bedrooms']==(i+2)],
      linewidth=1, width=0.5)
    ax.set_ylabel('Per Room Rental Price')
    ax.set_xlabel('Number of Bathrooms')
    ax.set_ylim([min_rent, max_rent])
    plt.title("%s Student Rental - %s Bed" %(outcode, i+2))


def areaOverview(grouped,outcode):

  fig = plt.figure(figsize = (14,4))

  ax1 = fig.add_subplot(121)
  plt.bar(x=grouped['num_bedrooms'],height=grouped['proportion%'], width=0.7, alpha=0.8)

  bars = range(grouped['num_bedrooms'].min(),grouped['num_bedrooms'].max()+1)
  plt.title
  plt.xticks(bars,bars)
  plt.xlabel('Number of Bedrooms')
  plt.ylabel('% of Listings')
  plt.title("%s Rental - How big are the houses" %(outcode))

  ax2 = fig.add_subplot(122)
  plt.scatter(x=grouped['num_bedrooms'],y=grouped['avg_baths'],
              c = grouped['proportion%'],marker='^', alpha=0.8)
  plt.xticks(bars,bars)
  plt.xlabel('Number of Bedrooms')
  plt.ylabel('Average Num. of Bathrooms')
  plt.title("%s Rental - Beds vs. Bathrooms" %(outcode))




