import sqlite3
import random
from datetime import datetime, timedelta

DATABASE = 'coffee_shop.db'

def insert_transaction(store_id, total_amount, timestamp):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Transactions (store_id, total_amount, timestamp) VALUES (?, ?, ?)",
                   (store_id, total_amount, timestamp))
    transaction_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return transaction_id

def insert_transaction_details(transaction_id, product_id, quantity, price):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO TransactionDetails (transaction_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
                   (transaction_id, product_id, quantity, price))
    connection.commit()
    connection.close()

def generate_past_data(start_date, end_date):
    current_date = start_date
    total_days = (end_date - start_date).days
    current_day = 0

    while current_date <= end_date:
        for store_id in [1001, 1002]:
            num_transactions = random.randint(5, 20)
            for _ in range(num_transactions):
                # Randomize hours and minutes
                timestamp = current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")  # Exclude seconds from the format

                total_amount = 0
                transaction_id = insert_transaction(store_id, total_amount, timestamp_str)
                
                num_products = random.randint(1, 5)
                for _ in range(num_products):
                    if store_id == 1001:
                        product_id = random.randint(2001, 2010)
                    elif store_id == 1002:
                        product_id = random.randint(2011, 2020)
                    quantity = random.randint(1, 3)
                    price = random.randint(2, 6)
                    total_amount += price * quantity
                    insert_transaction_details(transaction_id, product_id, quantity, price)
                
                # Update total amount for the transaction
                connection = sqlite3.connect(DATABASE)
                cursor = connection.cursor()
                cursor.execute("UPDATE Transactions SET total_amount = ? WHERE transaction_id = ?",
                               (total_amount, transaction_id))
                connection.commit()
                connection.close()
        
        current_date += timedelta(days=1)
        current_day += 1
        print(f"Progress: {current_day}/{total_days} days processed.")

# Set start and end dates for historical data
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 7, 6)

# Generate and insert past data
generate_past_data(start_date, end_date)

print("Past data inserted successfully.")
