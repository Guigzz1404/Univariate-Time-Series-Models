import pandas as pd
import datetime
import order_book_creation
import matplotlib.pyplot as plt
import seaborn as sns


# sns parameters
sns.set_theme(style="darkgrid")


def get_top_positions(order_book, n_positions):

    # Get the top bid
    buys = pd.DataFrame(order_book["Buy"].values())
    buys["Price"] = buys.Price.astype(float)/10000
    buys = buys.set_index("Price").sort_index(ascending=False)
    buys = buys.iloc[:n_positions].set_index("Timestamp", append=True)

    # Get the top ask
    sells = pd.DataFrame(order_book["Sell"].values())
    sells["Price"] = sells.Price.astype(float)/10000
    sells = sells.set_index("Price").sort_index()
    sells = sells.iloc[:n_positions].set_index("Timestamp", append=True)

    return buys, sells


def get_top_levels(order_book, n_levels):

    buys = pd.DataFrame(order_book["Buy"].values())
    buys["Price"] = buys.Price.astype(float)/10000
    bPriceList = buys["Price"].sort_values(ascending=False).unique()[:n_levels]
    buys = buys.set_index("Price").sort_index(ascending=False)
    buys = buys.loc[bPriceList].set_index("Timestamp", append=True)

    sells = pd.DataFrame(order_book["Sell"].values())
    sells["Price"] = sells.Price.astype(float)/10000
    sPriceList = sells["Price"].sort_values().unique()[:n_levels]
    sells = sells.set_index("Price").sort_index()
    sells = sells.loc[sPriceList].set_index("Timestamp", append=True)

    return buys, sells


# Creation of df
biof_data = pd.read_csv("biof_levelIII_data.csv")
origin = datetime.date.today()
biof_data["Timestamp"] = pd.to_datetime(biof_data["Timestamp"],unit='ms',origin=origin)
biof_data = biof_data.drop(columns=['Ticker','MPID'])
# print(biof_data.head())

order_book,trades = order_book_creation.process_book_data(biof_data, datetime.datetime.combine(origin,  datetime.time(11,0,0)))
buys, sells = get_top_levels(order_book, 10)

# We can now construct the consolidated order book
b = buys.groupby("Price")[["Quantity"]].sum().sort_index(ascending=False).reset_index() # [[]] to keep b in df type
s = sells.groupby("Price")[["Quantity"]].sum().sort_index().reset_index()

# Let's build the aggregated supply and demand curves
mid_price = (b['Price'].max()+s["Price"].min())/2

b['QtyCum_sum'] = b['Quantity'].cumsum()
b['QtyCum_perc'] = round(100*b.QtyCum_sum/b['Quantity'].sum(),2)
b['PriceDistance'] = mid_price - b['Price']
b['PriceCum_perc'] = round(-100*b.PriceDistance/b['PriceDistance'].max(),2)

s['QtyCum_sum'] = s['Quantity'].cumsum()
s['QtyCum_perc'] = round(100*s.QtyCum_sum/s['Quantity'].sum(),2)
s['PriceDistance'] = mid_price - s['Price']
s['PriceCum_perc'] = round(100*s.PriceDistance/s['PriceDistance'].min(),2)

# Let's plot the cumulative order book
# For visual cleanliness, we add a first row for the midpoint in the dataframes
s.loc[-1] = [mid_price, 0, 0, 0, 0, s.PriceCum_perc[0]]
s.index = s.index + 1
s = s.sort_index()
b.loc[-1] = [mid_price, 0, 0, 0, 0, b.PriceCum_perc[0]]
b.index = b.index + 1
b = b.sort_index()

fig,ax = plt.subplots(figsize=(12,8))
_ = plt.step(s['QtyCum_perc'], s['PriceCum_perc'], label='Sell side') # Stepped curve
_ = plt.step(b['QtyCum_perc'], b['PriceCum_perc'], label='Buy side')
_ = plt.ylabel('Price(%)',fontsize=16)
_ = plt.xlabel('Depth(%)',fontsize=16)
_ = plt.legend(loc='upper left', frameon=False,fontsize=16)
_ = plt.title('Cumulative Normalized Order Book',fontsize=16)
sns.despine()
plt.tight_layout()
plt.show()
