import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# MODE SWITCH

MODE = "streamlit"
if len(sys.argv) > 1 and sys.argv[1] == "cli":
    MODE = "cli"


# LOAD DATA

data = pd.read_csv("Mall_Customers.csv")


# PREPROCESSING

data["Discount Sensitivity"] = data["Discount Sensitivity"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Purchase Frequency",
    "Avg Purchase Value",
    "Recency (days)",
    "Website Visits",
    "Loyalty Score",
    "Discount Sensitivity"
]

y_cls = data["Will Purchase"]


# CLASSIFICATION MODEL

X_cls = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cls)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y_cls)


# REGRESSION MODEL

X_reg = data[["Annual Income (k$)", "Spending Score (1-100)"]]
y_reg = data["Annual Income (k$)"] * data["Spending Score (1-100)"] / 100

reg = LinearRegression()
reg.fit(X_reg, y_reg)


# CLUSTERING MODEL

X_cluster = data[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_cluster)

segment_names = {
    0: "LL",
    1: "HH",
    2: "HL",
    3: "LH",
    4: "Avg"
}


# STREAMLIT MODE

if MODE == "streamlit":
    import streamlit as st
    import matplotlib.pyplot as plt

    st.markdown("""
    # 🛒 Smart Customer Analytics System
    ### Predict Customer Behavior using Machine Learning
    """)

    st.subheader("📂 Dataset Preview")
    st.dataframe(data)

    st.subheader("🧾 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 25)
        income = st.slider("Annual Income (k$)", 10, 150, 50)
        score = st.slider("Spending Score", 1, 100, 50)
        purchase_freq = st.slider("Purchase Frequency", 1, 20, 5)

    with col2:
        avg_value = st.slider("Avg Purchase Value", 10, 500, 100)
        recency = st.slider("Recency (days)", 1, 365, 50)
        visits = st.slider("Website Visits", 1, 50, 10)
        loyalty = st.slider("Loyalty Score", 1, 100, 50)
        discount = st.selectbox("Discount Sensitivity", ["Low", "Medium", "High"])

    discount_map = {"Low": 0, "Medium": 1, "High": 2}
    discount = discount_map[discount]

    # Prediction
    user_input_cls = scaler.transform([[
        age, income, score,
        purchase_freq, avg_value,
        recency, visits,
        loyalty, discount
    ]])

    purchase_pred = clf.predict(user_input_cls)
    proba = clf.predict_proba(user_input_cls)

    spending_pred = reg.predict([[income, score]])
    spending_pred = max(spending_pred[0], 0)

    user_cluster = kmeans.predict([[income, score]])

    # Results
    st.divider()
    st.subheader("📊 Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Segment", segment_names[user_cluster[0]])
    c2.metric("Purchase Chance", f"{proba[0][1]*100:.1f}%")
    c3.metric("Spending", f"₹ {spending_pred:.2f}K")

    # Graph
    st.subheader("📈 Customer Segmentation Visualization")

    fig, ax = plt.subplots()

    ax.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        c=kmeans.labels_,
        cmap="viridis"
    )

    ax.scatter(income, score, color='red', s=200, label="You")

    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.legend()

    st.pyplot(fig)


#  CLI MODE

if MODE == "cli":
    print("\n--- CLI MODE ---")

    age = int(input("Age: "))
    income = int(input("Income (k$): "))
    score = int(input("Spending Score: "))
    purchase_freq = int(input("Purchase Frequency: "))
    avg_value = int(input("Avg Purchase Value: "))
    recency = int(input("Recency (days): "))
    visits = int(input("Website Visits: "))
    loyalty = int(input("Loyalty Score: "))
    discount = input("Discount Sensitivity (Low/Medium/High): ")

    discount_map = {"Low": 0, "Medium": 1, "High": 2}
    discount = discount_map[discount]

    user_input_cls = scaler.transform([[
        age, income, score,
        purchase_freq, avg_value,
        recency, visits,
        loyalty, discount
    ]])

    purchase_pred = clf.predict(user_input_cls)
    proba = clf.predict_proba(user_input_cls)

    spending_pred = reg.predict([[income, score]])
    spending_pred = max(spending_pred[0], 0)

    user_cluster = kmeans.predict([[income, score]])

    print("\n--- Results ---")
    print("Segment:", segment_names[user_cluster[0]])
    print("Purchase Chance:", round(proba[0][1]*100, 2), "%")

    if purchase_pred[0] == 1:
        print("Prediction: Likely to Buy")
    else:
        print("Prediction: Not Likely to Buy")

    print("Estimated Spending: ₹", round(spending_pred, 2), "K")
    import matplotlib.pyplot as plt

# Plot graph
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=kmeans.labels_,
    cmap="viridis"
)

# Highlight user point
plt.scatter(income, score, color='red', s=200, label="You")

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")

plt.legend()
plt.show()