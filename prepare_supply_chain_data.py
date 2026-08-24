"""
Supply Chain Data Preparation and Leakage-Free ML Risk Pipeline
Prepares:
1. cleaned_supply_chain_orders.csv (Fact table)
2. dim_order_risk.csv (ML Predictions and Risk Scoring)
3. dim_calendar.csv (Date Dimension table)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.utils import resample

def main():
    print("=" * 60)
    print("STEP 1: Loading and Cleaning DataCo Supply Chain Dataset")
    print("=" * 60)
    
    input_file = "DataCoSupplyChainDataset.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found in current directory.")
        
    df = pd.read_csv(input_file, encoding='latin-1')
    print(f"Raw shape: {df.shape}")
    print(f"Raw Unique Orders: {df['Order Id'].nunique()}")
    print(f"Raw Unique Order Items: {df['Order Item Id'].nunique()}")
    
    # Exclude Cancelled shipments
    df = df[~df['Delivery Status'].str.contains('Cancel', case=False, na=False)].copy()
    print(f"Cleaned shape (excl Cancelled): {df.shape}")
    print(f"Cleaned Unique Orders: {df['Order Id'].nunique()}")
    print(f"Cleaned Unique Order Items: {df['Order Item Id'].nunique()}")
    
    # Parse dates
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')
    
    # Delivery & Delay calculations
    df['Order Processing Time'] = (df['shipping date (DateOrders)'] - df['order date (DateOrders)']).dt.days
    df['Delay'] = df['Order Processing Time'] - df['Days for shipment (scheduled)']
    df['Is_Delayed'] = df['Delay'] > 0
    
    df['Delivery Performance'] = np.where(
        df['Delay'] > 0, "Delayed",
        np.where(df['Delay'] == 0, "On Time", "Early")
    )
    
    # Profitability Flag (clean without trailing/leading whitespace)
    df['Profitability Flag'] = np.where(
        df['Order Profit Per Order'] > 0, "Profit",
        np.where(df['Order Profit Per Order'] < 0, "Loss", "Break-even")
    )
    
    # Extract temporal helper columns
    df['Order_Date'] = df['order date (DateOrders)'].dt.strftime('%Y-%m-%d')
    df['Order_Year'] = df['order date (DateOrders)'].dt.year
    df['Order_Month'] = df['order date (DateOrders)'].dt.month
    df['Order_Month_Name'] = df['order date (DateOrders)'].dt.strftime('%b')
    df['Order_Day_Name'] = df['order date (DateOrders)'].dt.strftime('%a')
    df['Order_Hour'] = df['order date (DateOrders)'].dt.hour
    
    # Select cleanest column set for Fact table
    fact_cols = [
        'Order Item Id',
        'Order Id',
        'Order Customer Id',
        'Customer Segment',
        'Customer City',
        'Customer State',
        'Customer Country',
        'Department Name',
        'Category Name',
        'Product Name',
        'Order Region',
        'Market',
        'Order City',
        'Order State',
        'Order Country',
        'Shipping Mode',
        'Type',
        'Order Status',
        'Days for shipment (scheduled)',
        'Days for shipping (real)',
        'Order Processing Time',
        'Delay',
        'Is_Delayed',
        'Delivery Status',
        'Delivery Performance',
        'Sales',
        'Order Profit Per Order',
        'Profitability Flag',
        'order date (DateOrders)',
        'shipping date (DateOrders)',
        'Order_Date',
        'Order_Year',
        'Order_Month',
        'Order_Month_Name',
        'Order_Day_Name',
        'Order_Hour',
        'Order Item Discount',
        'Order Item Discount Rate',
        'Order Item Product Price',
        'Order Item Quantity'
    ]
    
    df_fact = df[fact_cols].copy()
    df_fact.to_csv('cleaned_supply_chain_orders.csv', index=False)
    print("Saved 'cleaned_supply_chain_orders.csv'")
    
    # =========================================================================
    # STEP 2: Leakage-Free Machine Learning Pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Training Leakage-Free Random Forest ML Model")
    print("=" * 60)
    
    feature_cols = [
        'Type', 
        'Days for shipment (scheduled)', 
        'Category Name', 
        'Customer Segment', 
        'Department Name', 
        'Order Region', 
        'Shipping Mode', 
        'Order_Month', 
        'Order_Hour'
    ]
    
    X = df[feature_cols].copy()
    y = df['Is_Delayed'].astype(int).values
    
    # Train-test split FIRST (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    cat_cols = ['Type', 'Category Name', 'Customer Segment', 'Department Name', 'Order Region', 'Shipping Mode']
    
    # Compute frequency encodings ONLY on X_train to prevent leakage
    freq_maps = {}
    for col in cat_cols:
        freq_maps[col] = (X_train[col].value_counts() / len(X_train)).to_dict()
    
    # Apply frequency encoding
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    X_full_enc = X.copy()
    
    for col in cat_cols:
        mapping = freq_maps[col]
        default_val = min(mapping.values()) if mapping else 0.0
        X_train_enc[col] = X_train[col].map(mapping).fillna(default_val)
        X_test_enc[col] = X_test[col].map(mapping).fillna(default_val)
        X_full_enc[col] = X[col].map(mapping).fillna(default_val)
    
    # Balance classes using resample on training set only
    train_df = X_train_enc.copy()
    train_df['target'] = y_train
    
    df_majority = train_df[train_df['target'] == 1]
    df_minority = train_df[train_df['target'] == 0]
    
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    
    train_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)
    X_train_bal = train_balanced.drop('target', axis=1)
    y_train_bal = train_balanced['target'].values
    
    print(f"Balanced training size: {len(X_train_bal)} rows")
    
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    
    # Evaluate on untouched test set
    y_pred = rf.predict(X_test_enc)
    y_prob_test = rf.predict_proba(X_test_enc)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob_test)
    
    print("\n--- Model Evaluation on Untouched Test Set (No Leakage) ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))
    
    # Predict risk probabilities on the entire dataset
    full_probs = rf.predict_proba(X_full_enc)[:, 1]
    full_preds = (full_probs >= 0.50).astype(int)
    
    # Create Risk Category
    risk_category = np.where(
        full_probs >= 0.75, "High Risk",
        np.where(full_probs >= 0.50, "Medium Risk", "Low Risk")
    )
    
    df_risk = pd.DataFrame({
        'Order Item Id': df['Order Item Id'],
        'Order Id': df['Order Id'],
        'Actual_Delayed': df['Is_Delayed'].astype(int),
        'Predicted_Delayed': full_preds,
        'Risk_Probability': np.round(full_probs, 4),
        'Risk_Category': risk_category
    })
    
    df_risk.to_csv('dim_order_risk.csv', index=False)
    print(f"Saved 'dim_order_risk.csv' with {len(df_risk)} rows.")
    print("Risk Category Breakdown:")
    print(df_risk['Risk_Category'].value_counts())
    
    # =========================================================================
    # STEP 3: Generate Calendar Dimension (Dim_Calendar)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Generating Dim_Calendar Table")
    print("=" * 60)
    
    min_date = df['order date (DateOrders)'].min().floor('D')
    max_date = df['order date (DateOrders)'].max().ceil('D')
    
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    df_calendar = pd.DataFrame({'Date': date_range})
    df_calendar['Date_Key'] = df_calendar['Date'].dt.strftime('%Y%m%d').astype(int)
    df_calendar['Date_Formatted'] = df_calendar['Date'].dt.strftime('%Y-%m-%d')
    df_calendar['Year'] = df_calendar['Date'].dt.year
    df_calendar['Quarter'] = df_calendar['Date'].dt.quarter
    df_calendar['Quarter_Name'] = 'Q' + df_calendar['Quarter'].astype(str)
    df_calendar['Month'] = df_calendar['Date'].dt.month
    df_calendar['Month_Name'] = df_calendar['Date'].dt.strftime('%b')
    df_calendar['Month_Year'] = df_calendar['Date'].dt.strftime('%b %Y')
    df_calendar['Day'] = df_calendar['Date'].dt.day
    df_calendar['Day_Of_Week'] = df_calendar['Date'].dt.dayofweek + 1  # 1=Mon, 7=Sun
    df_calendar['Day_Name'] = df_calendar['Date'].dt.strftime('%a')
    df_calendar['Is_Weekend'] = df_calendar['Day_Of_Week'].isin([6, 7]).astype(int)
    
    df_calendar.to_csv('dim_calendar.csv', index=False)
    print(f"Saved 'dim_calendar.csv' ({len(df_calendar)} dates from {min_date.date()} to {max_date.date()})")
    
    print("\n" + "=" * 60)
    print("ALL DATA PREPARATION & ML EXPORTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
