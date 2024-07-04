# main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import sqlite3
import uuid
from datetime import datetime, date
import random
import time
import threading

app = FastAPI()

DATABASE = 'store.db'

class Store(BaseModel):
    store_id: str
    name: str
    location: str
    opening_time: str
    closing_time: str

class Transaction(BaseModel):
    transaction_id: str
    store_id: str
    total_amount: float
    timestamp: str

class TransactionDetail(BaseModel):
    transaction_detail_id: str
    transaction_id: str
    product_id: str
    quantity: int
    price: float

class Forecast(BaseModel):
    forecast_id: str
    store_id: str
    forecast_date: str
    forecast_amount: float

class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    cost: float
    stock_level: int

class CashRegister(BaseModel):
    register_id: str
    store_id: str
    initial_cash: float
    current_cash: float

PRODUCTS = [
    {"product_id": "p1", "name": "Coffee", "category": "Beverages", "price": 5.0, "cost": 2.0},
    {"product_id": "p2", "name": "Latte", "category": "Beverages", "price": 6.0, "cost": 2.5},
    {"product_id": "p3", "name": "Croissant", "category": "Bakery", "price": 3.0, "cost": 1.0},
]

def insert_random_transaction(store_id):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    transaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    total_amount = 0
    details = []

    num_items = random.randint(1, 5)
    for _ in range(num_items):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 3)
        total_amount += product["price"] * quantity
        detail = {
            "transaction_detail_id": str(uuid.uuid4()),
            "transaction_id": transaction_id,
            "product_id": product["product_id"],
            "quantity": quantity,
            "price": product["price"]
        }
        details.append(detail)

    cursor.execute("INSERT INTO Transactions (transaction_id, store_id, total_amount, timestamp) VALUES (?, ?, ?, ?)",
                   (transaction_id, store_id, total_amount, timestamp))
    for detail in details:
        cursor.execute("INSERT INTO TransactionDetails (transaction_detail_id, transaction_id, product_id, quantity, price) VALUES (?, ?, ?, ?, ?)",
                       (detail["transaction_detail_id"], detail["transaction_id"], detail["product_id"], detail["quantity"], detail["price"]))
    connection.commit()
    connection.close()

def simulate_transactions(store_id):
    while True:
        insert_random_transaction(store_id)
        time.sleep(random.uniform(5, 30))  # Simulate transactions every 5 to 30 seconds

@app.post("/store/open")
def open_store(store: Store, background_tasks: BackgroundTasks):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Stores (store_id, name, location, opening_time, closing_time) VALUES (?, ?, ?, ?, ?)",
                   (store.store_id, store.name, store.location, store.opening_time, store.closing_time))
    connection.commit()
    connection.close()
    
    background_tasks.add_task(simulate_transactions, store.store_id)  # Start simulating transactions
    return {"message": "Store opened successfully"}

@app.post("/sales/forecast")
def set_forecast(forecast: Forecast):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT OR REPLACE INTO Forecasts (store_id, forecast_date, forecast_amount) VALUES (?, ?, ?)",
                   (forecast.store_id, forecast.forecast_date, forecast.forecast_amount))
    connection.commit()
    connection.close()
    return {"message": "Forecast set successfully"}

@app.get("/sales/forecast")
def get_forecast(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT forecast_amount FROM Forecasts WHERE store_id = ? AND forecast_date = ?",
                   (store_id, date.today()))
    forecast = cursor.fetchone()
    connection.close()
    if forecast:
        return {"forecast_amount": forecast[0]}
    else:
        raise HTTPException(status_code=404, detail="Forecast not found")

@app.get("/product/menu")
def get_product_menu():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Products")
    products = cursor.fetchall()
    connection.close()
    return {"products": products}

@app.get("/cash_register/status")
def get_cash_register_status(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT initial_cash, current_cash FROM CashRegister WHERE store_id = ?", (store_id,))
    cash_register = cursor.fetchone()
    connection.close()
    if cash_register:
        return {"initial_cash": cash_register[0], "current_cash": cash_register[1]}
    else:
        raise HTTPException(status_code=404, detail="Cash register not found")

@app.post("/transactions/record")
def record_transaction(transaction: Transaction, details: list[TransactionDetail]):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Transactions (transaction_id, store_id, total_amount, timestamp) VALUES (?, ?, ?, ?)",
                   (transaction.transaction_id, transaction.store_id, transaction.total_amount, transaction.timestamp))
    for detail in details:
        cursor.execute("INSERT INTO TransactionDetails (transaction_detail_id, transaction_id, product_id, quantity, price) VALUES (?, ?, ?, ?, ?)",
                       (detail.transaction_detail_id, detail.transaction_id, detail.product_id, detail.quantity, detail.price))
    connection.commit()
    connection.close()
    return {"message": "Transaction recorded successfully"}

@app.get("/transactions/realtime")
def get_realtime_transactions(store_id: str, from_time: str = None, to_time: str = None):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    query = "SELECT * FROM Transactions WHERE store_id = ?"
    params = [store_id]
    if from_time:
        query += " AND timestamp >= ?"
        params.append(from_time)
    if to_time:
        query += " AND timestamp <= ?"
        params.append(to_time)
    cursor.execute(query, params)
    transactions = cursor.fetchall()
    connection.close()
    return {"transactions": transactions}

@app.get("/sales/analytics")
def get_sales_analytics(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT SUM(total_amount) FROM Transactions WHERE store_id = ?", (store_id,))
    total_sales = cursor.fetchone()[0] or 0
    cursor.execute("SELECT forecast_amount FROM Forecasts WHERE store_id = ? AND forecast_date = ?", 
                   (store_id, date.today()))
    forecast = cursor.fetchone()
    forecast_amount = forecast[0] if forecast else 0
    target_remaining = max(0, forecast_amount - total_sales)
    cursor.execute("SELECT category, SUM(price * quantity) FROM TransactionDetails JOIN Products ON TransactionDetails.product_id = Products.product_id WHERE transaction_id IN (SELECT transaction_id FROM Transactions WHERE store_id = ?) GROUP BY category", 
                   (store_id,))
    sales_by_category = cursor.fetchall()
    connection.close()
    return {"total_sales": total_sales, "target_remaining": target_remaining, "sales_by_category": sales_by_category}

@app.get("/customer_traffic/realtime")
def get_customer_traffic(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM CustomerTraffic WHERE store_id = ? AND entry_timestamp >= ?", 
                   (store_id, datetime.now().date().isoformat()))
    customer_count = cursor.fetchone()[0]
    connection.close()
    return {"customer_count": customer_count}

from collections import defaultdict
from itertools import combinations

class MarketBasketAnalysis:
    def __init__(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transaction_data = defaultdict(list)

    def add_transaction(self, transaction_id, products):
        self.transaction_data[transaction_id] = products

    def generate_recommendations(self, store_id):
        # Step 1: Generate frequent itemsets
        frequent_itemsets = self._generate_frequent_itemsets()

        # Step 2: Generate association rules
        association_rules = self._generate_association_rules(frequent_itemsets)

        # Step 3: Filter rules by minimum confidence
        filtered_rules = self._filter_rules_by_confidence(association_rules)

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(filtered_rules, store_id)

        return recommendations

    def _generate_frequent_itemsets(self):
        # Initialize frequent itemsets with single items
        frequent_itemsets = defaultdict(int)
        for transaction_id, products in self.transaction_data.items():
            for product in products:
                frequent_itemsets[(product,)] += 1

        # Iterate through itemsets of increasing size
        k = 2
        while True:
            candidate_itemsets = self._generate_candidate_itemsets(frequent_itemsets, k)
            frequent_itemsets_k = defaultdict(int)
            for transaction_id, products in self.transaction_data.items():
                for itemset in candidate_itemsets:
                    if set(itemset).issubset(set(products)):
                        frequent_itemsets_k[itemset] += 1
            if not frequent_itemsets_k:
                break
            frequent_itemsets.update(frequent_itemsets_k)
            k += 1

        return frequent_itemsets

    def _generate_candidate_itemsets(self, frequent_itemsets, k):
        candidate_itemsets = set()
        for itemset1 in frequent_itemsets:
            for itemset2 in frequent_itemsets:
                if len(set(itemset1) & set(itemset2)) == k - 1:
                    candidate_itemset = tuple(sorted(set(itemset1) | set(itemset2)))
                    if len(candidate_itemset) == k:
                        candidate_itemsets.add(candidate_itemset)
        return candidate_itemsets

    def _generate_association_rules(self, frequent_itemsets):
        association_rules = []
        for itemset in frequent_itemsets:
            for k in range(1, len(itemset)):
                for antecedent in combinations(itemset, k):
                    consequent = tuple(set(itemset) - set(antecedent))
                    support = frequent_itemsets[itemset]
                    confidence = support / frequent_itemsets[antecedent]
                    association_rules.append((antecedent, consequent, support, confidence))
        return association_rules

    def _filter_rules_by_confidence(self, association_rules):
        filtered_rules = []
        for rule in association_rules:
            if rule[3] >= self.min_confidence:
                filtered_rules.append(rule)
        return filtered_rules

    def _generate_recommendations(self, filtered_rules, store_id):
        recommendations = []
        for rule in filtered_rules:
            antecedent, consequent, support, confidence = rule
            recommendation = {
                "product_combination": list(antecedent) + list(consequent),
                "lift": confidence / (support / len(self.transaction_data)),
                "confidence": confidence,
                "support": support
            }
            recommendations.append(recommendation)
        return recommendations

@app.get("/mba/recommendations")
def get_mba_recommendations(store_id: str):
    mba = MarketBasketAnalysis(min_support=0.01, min_confidence=0.5)
    for transaction_id, products in get_transactions(store_id).items():
        mba.add_transaction(transaction_id, products)
    recommendations = mba.generate_recommendations(store_id)
    return {"recommendations": recommendations}

def get_transactions(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT transaction_id, product_id FROM TransactionDetails WHERE store_id = ?", (store_id,))
    transactions = cursor.fetchall()
    connection.close()
    transaction_data = defaultdict(list)
    for transaction_id, product_id in transactions:
        transaction_data[transaction_id].append(product_id)
    return transaction_data

@app.get("/inventory/replenishment")
def get_inventory_replenishment(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT product_id, stock_level FROM Products WHERE stock_level < ?", (10,))
    low_stock_products = cursor.fetchall()
    connection.close()
    return {"low_stock_products": low_stock_products}

@app.get("/customer_feedback/realtime")
def get_customer_feedback(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM CustomerFeedback WHERE store_id = ?", (store_id,))
    feedback = cursor.fetchall()
    connection.close()
    return {"feedback": feedback}

@app.get("/sales_trends")
def get_sales_trends(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT strftime('%Y-%m', timestamp) as month, SUM(total_amount) as total_sales FROM Transactions WHERE store_id = ? GROUP BY month", (store_id,))
    sales_trends = cursor.fetchall()
    connection.close()
    return {"sales_trends": sales_trends}

@app.post("/store/close")
def close_store(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Transactions WHERE store_id = ?", (store_id,))
    transactions = cursor.fetchall()
    cursor.execute("SELECT SUM(total_amount) FROM Transactions WHERE store_id = ?", (store_id,))
    total_sales = cursor.fetchone()[0] or 0
    connection.close()
    return {"message": "Store closed successfully", "total_sales": total_sales, "transactions": transactions}

@app.get("/inventory/end_of_day_report")
def get_end_of_day_inventory_report(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Products WHERE stock_level < ?", (10,))
    low_stock_products = cursor.fetchall()
    connection.close()
    return {"low_stock_products": low_stock_products}

@app.get("/sales/daily_summary")
def get_daily_sales_summary(store_id: str):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("SELECT SUM(total_amount) FROM Transactions WHERE store_id = ? AND timestamp >= ? AND timestamp <= ?", 
                   (store_id, datetime.combine(date.today(), datetime.min.time()).isoformat(), datetime.combine(date.today(), datetime.max.time()).isoformat()))
    daily_sales = cursor.fetchone()[0] or 0
    cursor.execute("SELECT forecast_amount FROM Forecasts WHERE store_id = ? AND forecast_date = ?", 
                   (store_id, date.today()))
    forecast = cursor.fetchone()
    forecast_amount = forecast[0] if forecast else 0
    performance = (daily_sales / forecast_amount) * 100 if forecast_amount > 0 else 0
    connection.close()
    return {"daily_sales": daily_sales, "forecast_amount": forecast_amount, "performance_percentage": performance}

# To run the FastAPI app
# uvicorn main:app --reload
