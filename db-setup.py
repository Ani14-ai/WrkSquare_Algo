import sqlite3
import random
import datetime

DATABASE = 'coffee_shop.db'

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

def insert_past_transactions():
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        # Fetch the maximum existing transaction_id
        cursor.execute("SELECT MAX(transaction_id) FROM Transactions")
        max_transaction_id = cursor.fetchone()[0]
        transaction_id = max_transaction_id if max_transaction_id is not None else 4000
        
        # Define the start and end dates
        start_date = datetime.datetime(2024, 3, 1)
        end_date = datetime.datetime(2024, 7, 5)
        delta = datetime.timedelta(days=1)
        
        while start_date <= end_date:
            # Simulate multiple transactions per day
            num_transactions = random.randint(5, 15)
            for _ in range(num_transactions):
                transaction_id += 1
                store_id = 1001  # Use store_id 1001
                total_amount = round(random.uniform(10.0, 50.0), 2)
                timestamp = start_date + datetime.timedelta(minutes=random.randint(0, 1440))
                
                cursor.execute("INSERT INTO Transactions (transaction_id, store_id, total_amount, timestamp) VALUES (?, ?, ?, ?)",
                               (transaction_id, store_id, total_amount, timestamp.strftime("%Y-%m-%d %H:%M")))
                
                product_id = random.choice(range(2001, 2011))
                quantity = random.randint(1, 5)
                price = round(random.uniform(1.0, 20.0), 2)
                
                cursor.execute("INSERT INTO TransactionDetails (transaction_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
                               (transaction_id, product_id, quantity, price))
            
            start_date += delta
        
        connection.commit()
        connection.close()
        print("Past transactions inserted successfully.")
    except Exception as e:
        print(f"Error inserting past transactions: {str(e)}")

if __name__ == "__main__":
    insert_past_transactions()
