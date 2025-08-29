import pandas as pd
import datetime
import order_book_creation


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

print(buys.head())
print(sells.head())

