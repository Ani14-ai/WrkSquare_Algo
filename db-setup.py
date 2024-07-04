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
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
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
    cursor.executemany("INSERT INTO Products (product_id, name, category, price) VALUES (?, ?, ?, ?)",
                       products)
    connection.commit()
    connection.close()

# Create tables
create_tables()

# Inserting store details
store_id = 1001
store_name = 'Coffee Haven'
store_location = 'Downtown'
store_opening_time = '07:00'
store_closing_time = '23:00'
insert_store(store_id, store_name, store_location, store_opening_time, store_closing_time)

# Inserting products
products = [
    (2001, 'Cappuccino', 'Beverage', 16.00),
    (2002, 'Latte', 'Beverage', 17.00),
    (2003, 'Hot Chocolate', 'Beverage', 16.00),
    (2004, 'Croissant', 'Food', 10.00),
    (2005, 'Sandwich', 'Food', 20.00),
    (2006, 'Americano', 'Beverage', 12.00),
    (2007, 'Mocha', 'Beverage', 18.00),
    (2008, 'Frappuccino', 'Beverage', 20.00),
    (2009, 'Muffin', 'Food', 10.00),
    (2010, 'Chai Tea Latte', 'Beverage', 16.00)
]
insert_products(products)

print("Store and products inserted successfully.")
