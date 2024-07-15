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
from itertools import combinations
from collections import defaultdict

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
    return 3000 

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


        
@app.get("/631ae94b-13a6-485e-a49d-79df59feb687", summary="Market Basket Analysis" , tags=["My Store"])
def get_market_basket_analysis(store_id: int):
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

        # Create a list of transactions with the items purchased
        transaction_items = df_merged.groupby('transaction_id')['name'].apply(list).tolist()

        # Generate all possible combinations of two items per transaction
        combinations_list = []
        for items in transaction_items:
            combinations_list.extend(combinations(items, 2))

        # Calculate support, confidence, and lift
        item_count = defaultdict(int)
        combo_count = defaultdict(int)
        for items in transaction_items:
            for item in items:
                item_count[item] += 1
            for combo in combinations(items, 2):
                combo_count[combo] += 1

        total_transactions = len(transaction_items)
        mba_data = []
        for combo, count in combo_count.items():
            item1, item2 = combo
            support = count / total_transactions
            confidence = count / item_count[item1]
            lift = confidence / (item_count[item2] / total_transactions)
            mba_data.append({
                'item1': item1,
                'item2': item2,
                'support': support,
                'confidence': confidence,
                'lift': lift
            })

        df_mba = pd.DataFrame(mba_data)

        plt.figure(figsize=(14, 10))
        sizes = 1000 * df_mba['lift']
        scatter = plt.scatter(df_mba['item1'], df_mba['item2'], s=sizes, alpha=0.6, c=df_mba['lift'], cmap='viridis')

        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Lift', rotation=270, labelpad=15)

        # Title and labels
        plt.title('Basket Analysis Map')
        plt.xlabel('Item 1')
        plt.ylabel('Item 2')

        # Reduce number of labels to avoid overlapping
        labels = df_mba.apply(lambda row: f"{row['item1']} - {row['item2']}", axis=1)
        for i, label in enumerate(labels):
            if df_mba['lift'][i] > df_mba['lift'].quantile(0.75):  # Only label the top 25% lifts
                plt.text(df_mba['item1'][i], df_mba['item2'][i], label, fontsize=8, ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.5))

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, va='top')
        plt.tight_layout()
        mba_heatmap_path = 'mba_heatmap.png'
        plt.savefig(mba_heatmap_path)
        plt.close()
        # Prepare data for the table
        table_data = []
        for _, row in df_mba.iterrows():
            table_data.append({
                'Basket': f"{row['item1']} - {row['item2']}",
                'Sum of Support Basket': f"{row['support']*100:.2f}%",
                'Sum of Confidence of Prod1': f"{row['confidence']*100:.2f}%",
                'Sum of Confidence of Prod2': f"{row['confidence']*100:.2f}%",
                'Sum of Lift': f"{row['lift']:.2f}"
            })
        return FileResponse(mba_heatmap_path, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/2360728c-406d-4b53-9e4f-416671f521a5", summary="MBA-TABLE-JSON" , tags=["My Store"])
def get_market_basket_analysis(store_id: int):
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

        # Create a list of transactions with the items purchased
        transaction_items = df_merged.groupby('transaction_id')['name'].apply(list).tolist()

        # Generate all possible combinations of two items per transaction
        combinations_list = []
        for items in transaction_items:
            combinations_list.extend(combinations(items, 2))

        # Calculate support, confidence, and lift
        item_count = defaultdict(int)
        combo_count = defaultdict(int)
        for items in transaction_items:
            for item in items:
                item_count[item] += 1
            for combo in combinations(items, 2):
                combo_count[combo] += 1

        total_transactions = len(transaction_items)
        mba_data = []
        for combo, count in combo_count.items():
            item1, item2 = combo
            support = count / total_transactions
            confidence = count / item_count[item1]
            lift = confidence / (item_count[item2] / total_transactions)
            mba_data.append({
                'item1': item1,
                'item2': item2,
                'support': support,
                'confidence': confidence,
                'lift': lift
            })

        df_mba = pd.DataFrame(mba_data)
        # Prepare data for the table
        table_data = []
        for _, row in df_mba.iterrows():
            table_data.append({
                'Basket': f"{row['item1']} - {row['item2']}",
                'Sum of Support Basket': f"{row['support']*100:.2f}%",
                'Sum of Confidence of Prod1': f"{row['confidence']*100:.2f}%",
                'Sum of Confidence of Prod2': f"{row['confidence']*100:.2f}%",
                'Sum of Lift': f"{row['lift']:.2f}"
            })
        return JSONResponse(content=table_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

class StoreSummary(BaseModel):
    StoreID: int
    LifetimeSales: float
    AverageDailySales: float
    ATV_AOV: float
    AvgItemPrice: float
    SalesToday: float
    WTDSales: float
    MTDSales: float
    QTDSales: float
    YTDSales: float
    YoYGrowth: float
    TopPerformingItems: List[dict]
    BottomPerformingItems: List[dict]

def get_store_data(store_id: int):
    conn = get_db_connection()
    df_transactions = pd.read_sql_query(f"SELECT * FROM Transactions WHERE store_id = {store_id}", conn)
    df_transaction_details = pd.read_sql_query(f"SELECT * FROM TransactionDetails WHERE transaction_id IN (SELECT transaction_id FROM Transactions WHERE store_id = {store_id})", conn)
    df_products = pd.read_sql_query(f"SELECT * FROM Products WHERE store_id = {store_id}", conn)
    conn.close()
    return df_transactions, df_transaction_details, df_products


@app.post("/summarize_store", response_model=StoreSummary)
def summarize_store(store_id: int):
    df_transactions, df_transaction_details, df_products = get_store_data(store_id)
    
    if df_transactions.empty or df_transaction_details.empty or df_products.empty:
        raise HTTPException(status_code=404, detail="Store data not found")

    # Convert timestamps to datetime objects
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

    # Calculate Lifetime Sales and round to two decimal places
    lifetime_sales = round(df_transactions['total_amount'].sum(), 2)
    
    # Calculate Average Daily Sales and round to two decimal places
    total_days = (df_transactions['timestamp'].max().date() - df_transactions['timestamp'].min().date()).days + 1
    average_daily_sales = round(lifetime_sales / total_days if total_days > 0 else 0, 2)

    # Calculate ATV/AOV (Average Transaction Value) and round to two decimal places
    total_transactions = df_transactions.shape[0]
    atv_aov = round(lifetime_sales / total_transactions if total_transactions > 0 else 0, 2)
    
    # Calculate Average Item Price and round to two decimal places
    total_items_sold = df_transaction_details['quantity'].sum()
    avg_item_price = round(df_transaction_details['price'].mean() if total_items_sold > 0 else 0, 2)
    
    # Calculate Sales Today and round to two decimal places
    today = datetime.datetime.now().date()
    sales_today = round(df_transactions[df_transactions['timestamp'].dt.date == today]['total_amount'].sum(), 2)
    
    # Calculate WTD Sales (Week-to-Date) and round to two decimal places
    start_of_week = today - datetime.timedelta(days=7)
    wtd_sales = round(df_transactions[df_transactions['timestamp'].dt.date >= start_of_week]['total_amount'].sum(), 2)
    
    # Calculate MTD Sales (Month-to-Date) and round to two decimal places
    start_of_month = today.replace(day=1)
    mtd_sales = round(df_transactions[df_transactions['timestamp'].dt.date >= start_of_month]['total_amount'].sum(), 2)
    
    # Calculate QTD Sales (Quarter-to-Date) and round to two decimal places
    quarter = (today.month - 1) // 3 + 1
    start_of_quarter = datetime.datetime(today.year, 3 * (quarter - 1) + 1, 1).date()
    qtd_sales = round(df_transactions[df_transactions['timestamp'].dt.date >= start_of_quarter]['total_amount'].sum(), 2)
    
    # Calculate YTD Sales (Year-to-Date) and round to two decimal places
    start_of_year = today.replace(month=1, day=1)
    ytd_sales = round(df_transactions[df_transactions['timestamp'].dt.date >= start_of_year]['total_amount'].sum(), 2)
    
    # Calculate YoY Growth (Year-over-Year) and round to two decimal places
    last_year_start = start_of_year.replace(year=start_of_year.year - 1)
    last_year_end = start_of_year - datetime.timedelta(days=1)
    last_year_sales = df_transactions[(df_transactions['timestamp'].dt.date >= last_year_start) & (df_transactions['timestamp'].dt.date <= last_year_end)]['total_amount'].sum()
    yoy_growth = round(((ytd_sales - last_year_sales) / last_year_sales) * 100 if last_year_sales > 0 else 0, 2)
    
    # Calculate Top Performing Items
    top_performing_items = df_transaction_details.groupby('product_id')['quantity'].sum().nlargest(3).reset_index()
    top_performing_items = top_performing_items.merge(df_products[['product_id', 'name']], on='product_id')
    top_performing_items = top_performing_items[['name', 'quantity']].to_dict(orient='records')
    
    # Calculate Bottom Performing Items
    bottom_performing_items = df_transaction_details.groupby('product_id')['quantity'].sum().nsmallest(3).reset_index()
    bottom_performing_items = bottom_performing_items.merge(df_products[['product_id', 'name']], on='product_id')
    bottom_performing_items = bottom_performing_items[['name', 'quantity']].to_dict(orient='records')
    
    return StoreSummary(
        StoreID=store_id,
        LifetimeSales=lifetime_sales,
        AverageDailySales=average_daily_sales,
        ATV_AOV=atv_aov,
        AvgItemPrice=avg_item_price,
        SalesToday=sales_today,
        WTDSales=wtd_sales,
        MTDSales=mtd_sales,
        QTDSales=qtd_sales,
        YTDSales=ytd_sales,
        YoYGrowth=yoy_growth,
        TopPerformingItems=top_performing_items,
        BottomPerformingItems=bottom_performing_items
    )
      
        
@app.post("/d3b8f374-89b5-4db5-8d98-1c29e2a1e9e5", summary="Augmented Analytics" , tags=["My Store"])
def get_analytics(
    store_id: int,
    prompt_request: Annotated[
        PromptRequest,
        Body(
            openapi_examples={
                "line_plot": {
                    "summary": "Line Plot",
                    "description": """Generate a line plot showing the monthly sales revenue for the past four months, highlighting significant peaks and dips, and annotate major events or promotions that could have impacted sales.
                    
Generate a line plot showing the trend of customer footfall over the past year. Highlight any significant peaks or troughs and correlate them with marketing campaigns or seasonal events.

Generate a line plot showing the monthly sales trend for the retail store over the past 5 months. Highlight any significant increases or decreases, and correlate these with special events and festivals in the UAE, such as Ramadan, Eid, and National Day.""",
                    "value": {
                        "prompt": "Show a line plot of daily sales trends with time on the x-axis and total sales on the y-axis. Use vibrant colors and add a legend."
                    },
                },
                "pie_chart": {
                    "summary": "Pie Chart",
                    "description": """ Create a pie chart showing the distribution of sales among different product categories for the latest month. Use a diverse palette of colors for each category, add labels with both the category names and their respective sales percentages, and include a legend for clarity.

Create an interactive pie chart showing the proportion of sales across various regions in the UAE for the last quarter. Use a range of colors to represent different regions, display percentage labels on each slice, and include a legend with detailed region names.""",
                    "value": {
                        "prompt": "Create a pie chart showing the distribution of sales among different product categories. Use a variety of colors for each category."
                    },
                },
                "heatmap": {
                    "summary": "Heatmap",
                    "description": """Produce a heatmap of product popularity based on customer ratings and purchase frequency, highlighting products with high ratings and frequent purchases. Use product names on the y axis

Produce a heatmap displaying hourly sales data for each product on the latest date. Use a spectrum color scheme to indicate sales volume. Highlight the hours with peak sales with a distinct color and include product names on the y-axis 

Using today's sales data, perform a market basket analysis and generate a heatmap using Seaborn to show the most frequently bought together product combinations. Highlight the combinations that can help achieve a target sales of 3000 AED based on the latest purchasing patterns""",
                    "value": {
                        "prompt": "Generate a heatmap to show the sales amount for each product over time. Use a gradient color scheme to represent sales volume."
                    },
                },
                 "box_chart": {
                    "summary": "Box Plot",
                    "description": """ Create a visually appealing box plot to display the range of daily sales for a selected month. Each box should represent a different day of the month. Use a vibrant and cohesive color scheme to enhance visual appeal. Clearly highlight the interquartile range (IQR) and any outliers to provide insightful analysis. Use a legend

Create a box plot to showcase the distribution of daily sales over the past quarter. Use varied colors for each month and include a legend to clearly indicate which color corresponds to each month. Highlight the median sales value with a bold line for emphasis.""",
                    "value": {
                        "prompt": "Display a box plot of daily sales for each day in the month of June"
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
        graph_path = "/home/waysahead/sites/WrkSquare_Algo/exports/charts/temp_chart.png"
        if all(ord(char) < 128 for char in response):
            headers = {"AI-response": response}
        else:
            headers = {}
        if os.path.exists(graph_path):
            return FileResponse(graph_path, media_type="image/png", headers=headers)
        else:
            return JSONResponse(content={"AI-response": response})
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
