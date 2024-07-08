from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, Response, Query , Body
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
from typing import Annotated
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI( title='Retail Store Point-Of-Sales Analytics',
    description='Store APIs')

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

class PromptRequest(BaseModel):
    prompt: str
    


def get_forecast_amount():
    return 1000 

def get_cash_register_amount():
    return round(random.uniform(50, 200), 2)  

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

transaction_simulation_tasks = {}  
db_lock = threading.Lock()

@app.post("/e6a9fbd7-f487-4a47-bfa4-1d207b4d5686", summary="Open Store" , tags=["My Store"] )
def open_store(background_tasks: BackgroundTasks, store_id: int = Form(...)):
    try:
        forecast_amount = get_forecast_amount()
        cash_amount = get_cash_register_amount()
        
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        
        cursor.execute("INSERT OR REPLACE INTO Forecasts (store_id, forecast_amount) VALUES (?, ?)", (store_id, forecast_amount))
        cursor.execute("INSERT OR REPLACE INTO CashRegister (store_id, cash_amount) VALUES (?, ?)", (store_id, cash_amount))
        
        cursor.execute("SELECT * FROM Products WHERE store_id = ?", (store_id,))
        products = cursor.fetchall()
        
        product_menu = [{"product_id": product[0], "name": product[2], "category": product[3], "price": product[4]} for product in products]
        
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
        time.sleep(random.uniform(30,60)) 
        
@app.get("/f91e0ab8-4a7e-4e6f-95c2-f3f67c5a62c8", summary="Realtime Transactions" , tags=["My Store"])
def get_realtime_transactions(store_id: int):
    try:
        connection = create_connection(DATABASE)
        cursor = connection.cursor()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ? AND DATE(timestamp) = ?", (store_id, current_date))
        transactions = cursor.fetchall()
        
        connection.close()
        
        return [{"transaction_id": transaction[0], "store_id": transaction[1], "total_amount": transaction[2], "timestamp": transaction[3]} for transaction in transactions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
def generate_combined_graph(store_id: int, transactions: list):
    # Create graph (sales progress)
    df_transactions = pd.DataFrame(transactions, columns=['transaction_id', 'store_id', 'total_amount', 'timestamp'])
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'], format="%Y-%m-%d %H:%M")
    df_transactions.set_index('timestamp', inplace=True)
    
    # Resample to daily sales
    daily_sales = df_transactions['total_amount'].resample('D').sum()
    
    # Calculate progress towards 1000 AED
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    progress_target = 1000
    progress_data = pd.Series([0, progress_target], index=pd.date_range(start=current_date, periods=2, freq='D'))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=daily_sales, marker='o', color='b', label='Daily Sales')
    plt.plot(progress_data.index, [progress_target] * len(progress_data.index), linestyle='--', color='r', label=f'Progress towards {progress_target} AED')
    
    plt.title(f'Daily Sales Progress for Store {store_id} up to {current_date}')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (AED)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save graph
    combined_graph_path = f'combined_chart_{store_id}.png'
    plt.savefig(combined_graph_path)
    plt.close()
    
    return combined_graph_path
        
@app.post("/d3b8f374-89b5-4db5-8d98-1c29e2a1e9e5", summary="Augmented Analytics" , tags=["My Store"])
def get_analytics(
    store_id: int,
    prompt_request: Annotated[
        PromptRequest,
        Body(
            openapi_examples={
                "line_plot": {
                    "summary": "Line Plot",
                    "description": "Show a line plot of daily sales trends with time on the x-axis and total sales on the y-axis. Use vibrant colors and add a legend.",
                    "value": {
                        "prompt": "Show a line plot of daily sales trends with time on the x-axis and total sales on the y-axis. Use vibrant colors and add a legend."
                    },
                },
                "pie_chart": {
                    "summary": "Pie Chart",
                    "description": "Create a pie chart showing the distribution of sales among different product categories. Use a variety of colors for each category.",
                    "value": {
                        "prompt": "Create a pie chart showing the distribution of sales among different product categories. Use a variety of colors for each category."
                    },
                },
                "heatmap": {
                    "summary": "Heatmap",
                    "description": "Generate a heatmap to show the sales amount for each product over time. Use a gradient color scheme to represent sales volume.",
                    "value": {
                        "prompt": "Generate a heatmap to show the sales amount for each product over time. Use a gradient color scheme to represent sales volume."
                    },
                },
                "bar_chart": {
                    "summary": "Bar Chart",
                    "description": "Display a bar chart of the top 5 selling products with product names on the x-axis and sales amount on the y-axis. Use different colors for each bar.",
                    "value": {
                        "prompt": "Display a bar chart of the top 5 selling products with product names on the x-axis and sales amount on the y-axis. Use different colors for each bar."
                    },
                },
            },
        ),
    ]
):
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
        
        cursor.execute("SELECT * FROM Products WHERE store_id = ?", (store_id,))
        products = cursor.fetchall()
        
        df_products = pd.DataFrame(products, columns=['product_id', 'store_id', 'name', 'category', 'price'])
        
        # Merge DataFrames
        df_merged = pd.merge(df_transaction_details, df_products, on='product_id')
        df_merged = pd.merge(df_merged, df_transactions, on='transaction_id')
        
        # Create SmartDatalake
        lake = SmartDatalake([df_merged], config={"llm": llm})
        response = lake.chat(prompt_request.prompt + " All prices should be in AED")
        
        # Generate combined graph with subplots
        combined_graph_path = generate_combined_graph(store_id, transactions)
        
        # Path to the stored graph
        stored_graph_path = "/home/waysahead/sites/WrkSquare_Algo/exports/charts/temp_chart.png"
        
        # Create a new figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Load and plot the progress graph
        progress_img = plt.imread(combined_graph_path)
        axes[0].imshow(progress_img)
        axes[0].axis('off')
        axes[0].set_title('Progress Graph towards 1000 AED')
        
        # Load and plot the stored graph
        stored_img = plt.imread(stored_graph_path)
        axes[1].imshow(stored_img)
        axes[1].axis('off')
        axes[1].set_title('AI Graph')
        
        # Save combined graph
        combined_graph_with_stored_path = f'combined_with_stored_chart_{store_id}.png'
        plt.savefig(combined_graph_with_stored_path)
        plt.tight_layout()
        plt.close()
        
        headers = {"AI-response": response}
        return FileResponse(combined_graph_with_stored_path, media_type='image/png', headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/a9174a9f-3e16-47a4-9bce-8c4e04c6895b", summary="Close Store" , tags=["My Store"])
def close_store(background_tasks: BackgroundTasks, store_id: int = Form(...)):
    try:
        global transaction_simulation_tasks

        if store_id in transaction_simulation_tasks:
            # Stop the ongoing transaction simulation task
            transaction_simulation_tasks.pop(store_id)

        connection = create_connection(DATABASE)
        cursor = connection.cursor()

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM Transactions WHERE store_id = ? AND DATE(timestamp) = ?", (store_id, current_date))
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
