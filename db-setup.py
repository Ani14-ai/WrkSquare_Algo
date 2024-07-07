import sqlite3

DATABASE = 'coffee_shop.db'

def create_tables():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Stores (
        store_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        opening_time TEXT NOT NULL,
        closing_time TEXT NOT NULL
    )
    ''')

    cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    product_id INTEGER PRIMARY KEY,
    store_id INTEGER,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    FOREIGN KEY(store_id) REFERENCES Stores(store_id)
);

    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Transactions (
        transaction_id INTEGER PRIMARY KEY,
        store_id INTEGER,
        total_amount REAL,
        timestamp TEXT,
        FOREIGN KEY(store_id) REFERENCES Stores(store_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TransactionDetails (
        transaction_detail_id INTEGER PRIMARY KEY,
        transaction_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        price REAL,
        FOREIGN KEY(transaction_id) REFERENCES Transactions(transaction_id),
        FOREIGN KEY(product_id) REFERENCES Products(product_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Forecasts (
        store_id INTEGER PRIMARY KEY,
        forecast_amount REAL,
        FOREIGN KEY(store_id) REFERENCES Stores(store_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CashRegister (
        store_id INTEGER PRIMARY KEY,
        cash_amount REAL,
        FOREIGN KEY(store_id) REFERENCES Stores(store_id)
    )
    ''')
    
    connection.commit()
    connection.close()

def insert_store(store_id, name, location, opening_time, closing_time):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Stores (store_id, name, location, opening_time, closing_time) VALUES (?, ?, ?, ?, ?)",
                   (store_id, name, location, opening_time, closing_time))
    connection.commit()
    connection.close()

def insert_products(products):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.executemany("INSERT INTO Products (product_id, store_id, name, category, price) VALUES (?, ?, ?, ?, ?)",
                       products)
    connection.commit()
    connection.close()

# Create tables
create_tables()

# Inserting store details
store_id_1 = 1001
store_name_1 = 'Cafe Dubai'
store_location_1 = 'Dubai'
store_opening_time_1 = '08:00'
store_closing_time_1 = '22:00'
insert_store(store_id_1, store_name_1, store_location_1, store_opening_time_1, store_closing_time_1)

store_id_2 = 1002
store_name_2 = 'McDonalds'
store_location_2 = 'Dubai'
store_opening_time_2 = '07:00'
store_closing_time_2 = '23:00'
insert_store(store_id_2, store_name_2, store_location_2, store_opening_time_2, store_closing_time_2)

# Inserting products
products_store_1 = [
    (2001, store_id_1, 'Cappuccino', 'Beverage', 16),
    (2002, store_id_1, 'Latte', 'Beverage', 17),
    (2003, store_id_1, 'Hot Chocolate', 'Beverage', 16),
    (2004, store_id_1, 'Croissant', 'Food', 10),
    (2005, store_id_1, 'Sandwich', 'Food', 20),
    (2006, store_id_1, 'Americano', 'Beverage', 12),
    (2007, store_id_1, 'Mocha', 'Beverage', 18),
    (2008, store_id_1, 'Frappuccino', 'Beverage', 20),
    (2009, store_id_1, 'Muffin', 'Food', 10),
    (2010, store_id_1, 'Chai Tea Latte', 'Beverage', 16)
]
insert_products(products_store_1)


products_store_2 = [
    (2011, store_id_2, 'Big Mac', 'Burger', 10),
    (2012, store_id_2, 'Quarter Pounder with Cheese', 'Burger', 16),
    (2013, store_id_2, 'McChicken', 'Sandwich', 8),
    (2014, store_id_2, 'Filet-O-Fish', 'Sandwich', 5),
    (2015, store_id_2, 'French Fries', 'Side', 8),
    (2016, store_id_2, 'McFlurry', 'Dessert', 10),
    (2017, store_id_2, 'Apple Pie', 'Dessert', 5),
    (2018, store_id_2, 'Big Breakfast', 'Breakfast', 10),
    (2019, store_id_2, 'Egg McMuffin', 'Breakfast', 8),
    (2020, store_id_2, 'Sausage McMuffin with Egg', 'Breakfast', 10)
]
insert_products(products_store_2)


print("Stores and products inserted successfully.")
