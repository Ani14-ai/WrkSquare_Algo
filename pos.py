from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, Response, Query
from pydantic import BaseModel
import sqlite3
import random
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandasai import SmartDatalake
from pandasai.llm import OpenAI
from pandasai import Agent
from io import BytesIO
import base64
import threading
from fastapi.responses import FileResponse

app = FastAPI()
DATABASE = 'coffee_shop.db'
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["openai_api_key"] = OPENAI_API_KEY

llm = OpenAI(api_token=OPENAI_API_KEY)

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
    return 1000  # Example forecast amount in AED

def get_cash_register_amount():
    return round(random.uniform(50, 200), 2)  # Example initial cash amount

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

transaction_simulation_tasks = {}  # To keep track of ongoing transaction simulations
db_lock = threading.Lock()

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
        
        task = background_tasks.add_task(start_transaction_simulation, store_id)
        transaction_simulation_tasks[store_id] = task
        
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
    while store_id in transaction_simulation_tasks:
        with db_lock:
            simulate_transaction(store_id)
        time.sleep(random.uniform(5, 10))  # Reduce transaction frequency

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
def get_analytics(store_id: int, prompt: str = Query(...)):
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ?", (store_id,))
        transactions = cursor.fetchall()
        
        df_transactions = pd.DataFrame(transactions, columns=['transaction_id', 'store_id', 'total_amount', 'timestamp'])
        df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'], format="%Y-%m-%d %H:%M")
        
        cursor.execute("SELECT * FROM TransactionDetails")
        transaction_details = cursor.fetchall()
        
        df_transaction_details = pd.DataFrame(transaction_details, columns=['transaction_detail_id', 'transaction_id', 'product_id', 'quantity', 'price'])
        
        cursor.execute("SELECT * FROM Products")
        products = cursor.fetchall()
        
        df_products = pd.DataFrame(products, columns=['product_id', 'name', 'category', 'price'])
        
        # Merge DataFrames
        df_merged = pd.merge(df_transaction_details, df_products, on='product_id')
        df_merged = pd.merge(df_merged, df_transactions, on='transaction_id')        
        # Use OpenAI API to interpret the prompt and generate response
        lake = SmartDatalake([df_merged], config={"llm": llm})
        response = lake.chat(prompt + "All prices should be in AED")
        graph_path = "/home/waysahead/sites/WrkSquare_Algo/exports/charts/temp_chart.png"
        header=response
        if os.path.exists(graph_path):
            return FileResponse(graph_path, media_type="image/png",headers=header)
        else:
             raise HTTPException(status_code=500, detail="Graph file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store/close")
def close_store(background_tasks: BackgroundTasks, store_id: int = Form(...)):
    try:
        global transaction_simulation_tasks
        
        if store_id in transaction_simulation_tasks:
            # Stop the ongoing transaction simulation task
            transaction_simulation_tasks.pop(store_id)
        
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
    uvicorn.run(app, host="0.0.0.0", port=5000)
