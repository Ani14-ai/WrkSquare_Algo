from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, Response
from pydantic import BaseModel
import sqlite3
import random
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import threading

from sqlite3 import Error

app = FastAPI()

DATABASE = 'coffee_shop.db'
transaction_simulation_tasks = {}  # To keep track of ongoing transaction simulations
db_lock = threading.Lock()

class Transaction(BaseModel):
    transaction_id: int
    store_id: int
    total_amount: float
    timestamp: str

class TransactionDetail(BaseModel):
    transaction_detail_id: int
    transaction_id: int
    product_id: int
    quantity: int
    price: float

def get_forecast_amount():
    return 1000  

def get_cash_register_amount():
    return round(random.uniform(50, 200), 2)  # Example initial cash amount

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except Error as e:
        print(e)
    return None

@app.post("/store/open")
def open_store(background_tasks: BackgroundTasks, store_id: int = Form(...)):
    try:
        forecast_amount = get_forecast_amount()
        cash_amount = get_cash_register_amount()
        
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("INSERT OR REPLACE INTO Forecasts (store_id, forecast_amount) VALUES (?, ?)", (store_id, forecast_amount))
        cursor.execute("INSERT OR REPLACE INTO CashRegister (store_id, cash_amount) VALUES (?, ?)", (store_id, cash_amount))
        
        cursor.execute("SELECT * FROM Products")
        products = cursor.fetchall()
        
        product_menu = [{"product_id": product[0], "name": product[1], "category": product[2], "price": product[3]} for product in products]
        
        connection.commit()
        connection.close()
        
        background_tasks.add_task(start_transaction_simulation, store_id)
        
        return {
            "message": "Store opened successfully",
            "forecast_amount": forecast_amount,
            "cash_register_amount": cash_amount,
            "product_menu": product_menu
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def simulate_transaction(store_id: int):
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("SELECT MAX(transaction_id) FROM Transactions")
        max_transaction_id = cursor.fetchone()[0]
        transaction_id = 4001 if max_transaction_id is None else max_transaction_id + 1
        
        total_amount = round(random.choice([16.00, 17.00, 10.00, 20.00, 12.00, 18.00]), 2)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        cursor.execute("INSERT INTO Transactions (transaction_id, store_id, total_amount, timestamp) VALUES (?, ?, ?, ?)",
                       (transaction_id, store_id, total_amount, timestamp))
        
        product_id = random.choice(range(2001, 2011))
        quantity = random.randint(1, 3)
        price = cursor.execute("SELECT price FROM Products WHERE product_id = ?", (product_id,)).fetchone()[0]
        price = round(price * quantity, 2)
        
        cursor.execute("INSERT INTO TransactionDetails (transaction_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
                       (transaction_id, product_id, quantity, price))
        
        connection.commit()
        connection.close()
    except Exception as e:
        print(f"Error simulating transaction: {str(e)}")

def start_transaction_simulation(store_id: int):
    while True:
        with db_lock:
            simulate_transaction(store_id)
        time.sleep(random.uniform(1, 5))

@app.get("/transactions/realtime")
def get_realtime_transactions(store_id: int):
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ?", (store_id,))
        transactions = cursor.fetchall()
        
        connection.close()
        
        return [{"transaction_id": transaction[0], "store_id": transaction[1], "total_amount": transaction[2], "timestamp": transaction[3]} for transaction in transactions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics(store_id: int):
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ?", (store_id,))
        transactions = cursor.fetchall()
        
        df_transactions = pd.DataFrame(transactions, columns=['transaction_id', 'store_id', 'total_amount', 'timestamp'])
        df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'], format='%Y-%m-%d %H:%M')
        
        # Basic sales analytics
        total_sales = df_transactions['total_amount'].sum()
        total_transactions = df_transactions.shape[0]
        avg_transaction_amount = df_transactions['total_amount'].mean()
        
        hourly_sales = df_transactions.groupby(df_transactions['timestamp'].dt.hour)['total_amount'].sum()
        avg_hourly_transactions = df_transactions.groupby(df_transactions['timestamp'].dt.hour).size().mean()
        
        # Most and least selling products
        cursor.execute("SELECT product_id, SUM(quantity) FROM TransactionDetails GROUP BY product_id ORDER BY SUM(quantity) DESC LIMIT 1")
        most_selling_product_id = cursor.fetchone()
        most_selling_product_id = most_selling_product_id[0] if most_selling_product_id else None
        
        cursor.execute("SELECT product_id, SUM(quantity) FROM TransactionDetails GROUP BY product_id ORDER BY SUM(quantity) ASC LIMIT 1")
        least_selling_product_id = cursor.fetchone()
        least_selling_product_id = least_selling_product_id[0] if least_selling_product_id else None
        
        cursor.execute("SELECT name FROM Products WHERE product_id = ?", (most_selling_product_id,))
        most_selling_product_name = cursor.fetchone()[0] if most_selling_product_id else None
        
        cursor.execute("SELECT name FROM Products WHERE product_id = ?", (least_selling_product_id,))
        least_selling_product_name = cursor.fetchone()[0] if least_selling_product_id else None
        
        # Generate subplots
        plt.figure(figsize=(15, 10))
        
        # Hourly Sales subplot
        plt.subplot(2, 1, 1)
        sns.barplot(x=hourly_sales.index, y=hourly_sales.values)
        plt.title('Hourly Sales')
        plt.xlabel('Hour')
        plt.ylabel('Sales Amount')
        
        # Sales Heatmap subplot
        plt.subplot(2, 1, 2)
        pivot_table = df_transactions.pivot_table(values='total_amount', index=df_transactions['timestamp'].dt.date, columns=df_transactions['timestamp'].dt.hour, aggfunc='sum')
        sns.heatmap(pivot_table, cmap="YlGnBu", annot=True)
        plt.title('Sales Heatmap')
        plt.xlabel('Hour')
        plt.ylabel('Date')
        
        # Save the figure to BytesIO
        combined_plot_img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(combined_plot_img, format='png')
        plt.close()
        combined_plot_img.seek(0)
        
        connection.close()
        
        # Return the combined plot as Response with appropriate headers
        headers = {
            "total_sales": str(total_sales),
            "total_transactions": str(total_transactions),
            "avg_transaction_amount": str(avg_transaction_amount),
            "avg_hourly_transactions": str(avg_hourly_transactions),
            "most_selling_product": most_selling_product_name,
            "least_selling_product": least_selling_product_name
        }
        
        return Response(content=combined_plot_img.getvalue(), media_type="image/png", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store/close")
def close_store(background_tasks: BackgroundTasks, store_id: int = Form(...)):
    try:
        global transaction_simulation_tasks
        
        if store_id in transaction_simulation_tasks:
            # Stop the ongoing transaction simulation task
            task = transaction_simulation_tasks.pop(store_id)
            task.cancel()
        
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ?", (store_id,))
        transactions = cursor.fetchall()
        
        df_transactions = pd.DataFrame(transactions, columns=['transaction_id', 'store_id', 'total_amount', 'timestamp'])
        df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'], format='%Y-%m-%d %H:%M')
        
        total_sales = df_transactions['total_amount'].sum()
        
        cursor.execute("SELECT forecast_amount FROM Forecasts WHERE store_id = ?", (store_id,))
        forecast_amount = cursor.fetchone()[0]
        
        target_achieved = bool(total_sales >= forecast_amount)
        
        connection.close()
        
        return {
            "message": "Store closed successfully",
            "total_sales": float(total_sales),
            "forecast_amount": float(forecast_amount),
            "target_achieved": target_achieved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
