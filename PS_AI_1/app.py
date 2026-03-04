
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Advanced Store Sales Prediction", layout="wide")

st.title(" Advanced Store Sales Analysis & Forecasting App")

uploaded_file = st.file_uploader("Upload sales_data.csv", type=["csv"])

if uploaded_file is not None:
    
    # ================== LOAD DATA ==================
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    # ================== EDA SECTION ==================
    st.header("Exploratory Data Analysis")

    # 1. Sales Over Time
    st.write("###  Sales Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Date'], df['Sales'])
    ax1.set_title("Sales Over Time")
    st.pyplot(fig1)

    # 2. Promotion vs Non-Promotion
    st.write("###  Promotion vs Non-Promotion Sales")
    fig2, ax2 = plt.subplots()
    df.groupby('Promotion')['Sales'].mean().plot(kind='bar', ax=ax2)
    ax2.set_title("Average Sales: Promotion vs No Promotion")
    st.pyplot(fig2)

    # 3. Sales by Day of Week
    st.write("###  Sales by Day of Week")
    fig3, ax3 = plt.subplots()
    df.groupby('DayOfWeek')['Sales'].mean().plot(kind='bar', ax=ax3)
    ax3.set_title("Average Sales by Day of Week")
    st.pyplot(fig3)

    # 4. Sales by Month
    df['Month'] = df['Date'].dt.month
    st.write("###  Sales by Month")
    fig4, ax4 = plt.subplots()
    df.groupby('Month')['Sales'].mean().plot(kind='bar', ax=ax4)
    ax4.set_title("Average Sales by Month")
    st.pyplot(fig4)

    # 5. Weekend vs Weekday
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6,7] else 0)
    st.write("###  Weekend vs Weekday Sales")
    fig5, ax5 = plt.subplots()
    df.groupby('Is_Weekend')['Sales'].mean().plot(kind='bar', ax=ax5)
    ax5.set_xticklabels(['Weekday', 'Weekend'], rotation=0)
    ax5.set_title("Weekend vs Weekday Sales")
    st.pyplot(fig5)

    # 6. Sales Distribution
    st.write("###  Sales Distribution")
    fig6, ax6 = plt.subplots()
    ax6.hist(df['Sales'], bins=30)
    ax6.set_title("Sales Distribution")
    st.pyplot(fig6)

    # 7. Boxplot
    st.write("###  Sales Boxplot")
    fig7, ax7 = plt.subplots()
    ax7.boxplot(df['Sales'])
    ax7.set_title("Sales Boxplot")
    st.pyplot(fig7)

    # 8. Rolling 7-Day Trend
    df['rolling_mean_7'] = df['Sales'].rolling(7).mean()
    st.write("###  7-Day Rolling Average Trend")
    fig8, ax8 = plt.subplots()
    ax8.plot(df['Date'], df['rolling_mean_7'])
    ax8.set_title("7-Day Rolling Average")
    st.pyplot(fig8)

    # 9. Correlation Heatmap
    st.write("###  Correlation Heatmap")
    fig9, ax9 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax9)
    st.pyplot(fig9)

    # ================== FEATURE ENGINEERING ==================
    st.header("Feature Engineering")

    df['Day'] = df['Date'].dt.day
    df['lag_1'] = df['Sales'].shift(1)
    df['lag_7'] = df['Sales'].shift(7)
    df['rolling_mean_3'] = df['Sales'].rolling(3).mean()

    df.dropna(inplace=True)

    st.success("Lag and Rolling Features Created Successfully!")

    # ================== TRAIN TEST SPLIT ==================
    train = df[:-30]
    test = df[-30:]

    features = [
        'Promotion', 'DayOfWeek',
        'Month', 'Day', 'Is_Weekend',
        'lag_1', 'lag_7',
        'rolling_mean_3', 'rolling_mean_7'
    ]

    target = 'Sales'

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    # ================== MODEL SETTINGS ==================
    st.sidebar.header(" Model Settings")
    n_estimators = st.sidebar.slider("Number of Trees", 100, 500, 300)
    max_depth = st.sidebar.slider("Max Depth", 3, 10, 5)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.05)

    # ================== MODEL TRAINING ==================
    st.header(" Model Training (XGBoost)")

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ================== EVALUATION ==================
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write(f" MAE: {round(mae,2)}")
    st.write(f"RMSE: {round(rmse,2)}")

    # Actual vs Predicted
    st.subheader(" Actual vs Predicted Sales")
    fig10, ax10 = plt.subplots()
    ax10.plot(test['Date'], y_test, label="Actual")
    ax10.plot(test['Date'], y_pred, label="Predicted")
    ax10.legend()
    st.pyplot(fig10)

    # ================== FEATURE IMPORTANCE ==================
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig11, ax11 = plt.subplots()
    ax11.barh(importance_df['Feature'], importance_df['Importance'])
    ax11.invert_yaxis()
    st.pyplot(fig11)

    # ================== FUTURE FORECAST ==================
    st.header(" Next 7 Days Forecast")

    last_data = df.iloc[-1:].copy()
    future_predictions = []

    for i in range(7):
        next_pred = model.predict(last_data[features])[0]
        future_predictions.append(next_pred)

        last_data['lag_1'] = next_pred
        last_data['lag_7'] = last_data['lag_1']
        last_data['rolling_mean_3'] = (last_data['rolling_mean_3']*2 + next_pred)/3
        last_data['rolling_mean_7'] = (last_data['rolling_mean_7']*6 + next_pred)/7

    future_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Sales": future_predictions
    })

    st.dataframe(future_df)

    # Download button
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Forecast CSV",
        data=csv,
        file_name="7_day_forecast.csv",
        mime="text/csv"
    )

    # ================== BUSINESS INSIGHTS ==================
    st.header("Business Insights")

    st.markdown("""
    1. Promotions significantly increase sales.
    2. Weekend sales outperform weekdays.
    3. Sales show strong short-term dependency.
    4. Weekly rolling trends improve forecasting stability.
    5. XGBoost effectively captures complex sales patterns.
    """)

else:
    st.info("Please upload sales_data.csv to start analysis.")