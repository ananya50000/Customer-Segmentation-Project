import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("superstore.csv", encoding="latin1")

print("Dataset Loaded Successfully")

# Customer Summary
customer_data = df.groupby("Customer ID").agg({
    "Sales": "sum",
    "Profit": "sum",
    "Quantity": "sum"
}).reset_index()

print(customer_data.head())

# Create Clusters
kmeans = KMeans(n_clusters=3, random_state=42)

customer_data["Cluster"] = kmeans.fit_predict(
    customer_data[["Sales", "Profit", "Quantity"]]
)

# Save Output
customer_data.to_csv(
    "customer_segments.csv",
    index=False
)

# Create Visualization
plt.figure(figsize=(8,6))

plt.scatter(
    customer_data["Sales"],
    customer_data["Profit"],
    c=customer_data["Cluster"]
)

plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Customer Segmentation")

plt.savefig("customer_segments.png")

plt.show()

print("Project Completed Successfully")
print("Generated Files:")
print("customer_segments.csv")
print("customer_segments.png")