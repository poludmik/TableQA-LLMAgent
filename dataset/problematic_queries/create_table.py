# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random
#
# # Define the number of rows
# num_rows = 100
#
# # Column names with specifications
# columns = [
#     "Transaction ID", "Date", "Time", "Amount_USD", "Amount_EUR", "Currency",
#     "Sender Account", "Receiver Account", "Transaction Type", "Location",
#     "Status Before", "Status After", "Status Final", "Fees USD", "Fees EUR", "Fees CZK",
#     "exchange_rate_eur_czk", "exchange_rate_usd_czk", "Authentication Method", "Customer ID", "Merchant ID",
#     "Transaction Platform", "Device Used", "Network Type", "IP Address",
#     "Country Code", "City", "Postal Code", "Region", "Region_k", "Time Zone",
#     "Latency ms", "Session ID", "Browser", "OS", "App Version", "Error Code",
#     "Resolution Attempt", "Follow-up Date", "Agent ID", "Resolution Status",
#     "Comments", "Notification Sent", "Read Receipt", "Modification Flag",
#     "Previous Transaction ID", "Linked Transaction ID", "Batch ID", "Batch Status",
#     "Processing Time", "queue_time_start", "queue_time_end", "Approval Status", "Approval ID"
# ]
#
# # Generate random data for each column
# data = {col: [] for col in columns}
# for _ in range(num_rows):
#     data["Transaction ID"].append(f"TX{np.random.randint(1, 1000000):06d}")
#     data["Date"].append((datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime("%Y-%m-%d"))
#     data["Time"].append(datetime.now().strftime("%H:%M:%S"))
#     data["Amount_USD"].append(round(np.random.uniform(50, 5000), 2))
#     data["Amount_EUR"].append(round(np.random.uniform(45, 4500), 2))
#     data["Currency"].append(np.random.choice(["USD", "EUR", "CZK"]))
#     data["Sender Account"].append(f"SA{np.random.randint(1, 1000000):08d}")
#     data["Receiver Account"].append(f"RA{np.random.randint(1, 1000000):08d}")
#     data["Transaction Type"].append(np.random.choice(["Deposit", "Withdrawal", "Transfer"]))
#     data["Location"].append(np.random.choice(["Online", "Branch", "ATM"]))
#     data["Status Before"].append(np.random.choice(["Pending", "Processed", "Cancelled"]))
#     data["Status After"].append(np.random.choice(["Pending", "Processed", "Cancelled"]))
#     data["Status Final"].append(np.random.choice(["Completed", "Failed", "Reversed"]))
#     data["Fees USD"].append(round(np.random.uniform(0, 100), 2))
#     data["Fees EUR"].append(round(np.random.uniform(0, 90), 2))
#     data["Fees CZK"].append(round(np.random.uniform(0, 2400), 2))
#     data["exchange_rate_eur_czk"].append(round(np.random.uniform(24, 27), 2))
#     data["exchange_rate_usd_czk"].append(round(np.random.uniform(20, 23), 2))
#     data["Authentication Method"].append(np.random.choice(["Password", "Fingerprint", "Face ID"]))
#     data["Customer ID"].append(f"CU{np.random.randint(1, 1000000):08d}")
#     data["Merchant ID"].append(f"ME{np.random.randint(1, 1000000):08d}")
#     data["Transaction Platform"].append(np.random.choice(["Web", "Mobile App", "Telephone"]))
#     data["Device Used"].append(np.random.choice(["iPhone", "Android Phone", "Desktop", "Tablet"]))
#     data["Network Type"].append(np.random.choice(["WiFi", "4G", "5G"]))
#     data["IP Address"].append(f"{np.random.randint(100, 200)}.{np.random.randint(100, 200)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}")
#     data["Country Code"].append(np.random.choice(["US", "GB", "DE", "CZ"]))
#     data["City"].append(np.random.choice(["New York", "London", "Berlin", "Prague"]))
#     data["Postal Code"].append(f"{np.random.randint(10000, 99999)}")
#     data["Region"].append(np.random.choice(["North", "South", "East", "West"]))
#     data["Region_k"] = [f"RK{np.random.randint(1000, 9999):04d}" for _ in range(num_rows)]
#     data["Time Zone"] = [random.choice(["UTC", "CET", "PST", "CST", "EST"]) for _ in range(num_rows)]
#     data["Latency ms"] = [np.random.randint(10, 500) for _ in range(num_rows)]
#     data["Session ID"] = [f"SID{np.random.randint(10000000, 99999999):08d}" for _ in range(num_rows)]
#     data["Browser"] = [random.choice(["Chrome", "Firefox", "Safari", "Edge"]) for _ in range(num_rows)]
#     data["OS"] = [random.choice(["Windows", "macOS", "Android", "iOS"]) for _ in range(num_rows)]
#     data["App Version"] = [f"{np.random.randint(1, 10)}.{np.random.randint(0, 9)}.{np.random.randint(0, 9)}" for _ in
#                            range(num_rows)]
#     data["Error Code"] = ["" if np.random.rand() > 0.1 else f"ERR-{np.random.randint(100, 999):03d}" for _ in
#                           range(num_rows)]
#     data["Resolution Attempt"] = [np.random.choice(["Yes", "No"]) for _ in range(num_rows)]
#     data["Follow-up Date"] = [
#         "" if np.random.rand() > 0.2 else (datetime.now() + timedelta(days=np.random.randint(1, 30))).strftime(
#             "%Y-%m-%d") for _ in range(num_rows)]
#     data["Agent ID"] = [f"AG{np.random.randint(1000, 9999):04d}" if np.random.rand() > 0.7 else "" for _ in
#                         range(num_rows)]
#     data["Resolution Status"] = [random.choice(["Pending", "Resolved", "Closed"]) for _ in range(num_rows)]
#     data["Comments"] = ["" if np.random.rand() > 0.3 else f"Random comment {np.random.randint(1, 100)}" for _ in
#                         range(num_rows)]
#     data["Notification Sent"] = [random.choice(["Yes", "No"]) for _ in range(num_rows)]
#     data["Read Receipt"] = [random.choice(["Yes", "No"]) for _ in range(num_rows)]
#     data["Modification Flag"] = [random.choice(["Yes", "No"]) for _ in range(num_rows)]
#     data["Previous Transaction ID"] = ["" if np.random.rand() > 0.5 else f"TX{np.random.randint(1, 1000000):06d}" for _
#                                        in range(num_rows)]
#     data["Linked Transaction ID"] = ["" if np.random.rand() > 0.6 else f"TX{np.random.randint(1, 1000000):06d}" for _ in
#                                      range(num_rows)]
#     data["Batch ID"] = [f"BID{np.random.randint(10000, 99999):05d}" for _ in range(num_rows)]
#     data["Batch Status"] = [random.choice(["Pending", "Completed", "Failed"]) for _ in range(num_rows)]
#     data["Processing Time"] = [f"{np.random.randint(10, 1000)} ms" for _ in range(num_rows)]
#     data["queue_time_start"] = [datetime.now().strftime("%H:%M:%S.%f")[:-3] for _ in
#                                 range(num_rows)]  # Truncate milliseconds to avoid precision issues
#     data["queue_time_end"] = [datetime.now().strftime("%H:%M:%S.%f")[:-3] for _ in range(num_rows)]
#     data["Approval Status"] = [random.choice(["Pending", "Approved", "Rejected"]) for _ in range(num_rows)]
#     data["Approval ID"] = [f"AID{np.random.randint(10000, 99999):05d}" for _ in range(num_rows)]
#
# # Create a DataFrame
# df = pd.DataFrame(data)
# # save the DataFrame to a xlsx file
# df.to_excel("transactions_fin.xlsx", index=False)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define the number of rows
num_rows = 100

# Define column names, transform to lower case with underscores
columns = [
    "vehicle_id", "model", "year", "purchase_date", "initial_mileage", "current_mileage",
    "maintenance_schedule", "year_month", "service_date", "first_service_date", "last_service_date",
    "service_type", "cost_of_service", "next_service_date", "operational_status",
    "fuel_consum_start", "fuel_consum_end", "driver_assigned", "length_mean", "length_total",
    "fuel_pos", "fuel_neg"
]

# Generate random data for each column
data = {
    "vehicle_id": [f"VH{np.random.randint(1000, 9999)}" for _ in range(num_rows)],
    "model": np.random.choice(["Model X", "Model Y", "Model S", "Model 3"], num_rows),
    "year": np.random.randint(1990, 2023, num_rows),
    "purchase_date": [(datetime.now() - timedelta(days=np.random.randint(0, 365 * 10))).date() for _ in range(num_rows)],
    "initial_mileage": np.random.randint(0, 50000, num_rows),
    "current_mileage": lambda initial_mileage: initial_mileage + np.random.randint(1000, 50000, num_rows),
    "maintenance_schedule": np.random.choice(["Annual", "Semi-annual", "Quarterly"], num_rows),
    "year_month": [int((datetime.now() - timedelta(days=np.random.randint(0, 365 * 2))).strftime("%Y%m")) for _ in range(num_rows)],
    "service_date": [(datetime.now() - timedelta(days=np.random.randint(0, 365))).date() for _ in range(num_rows)],
    "first_service_date": [(datetime.now() - timedelta(days=np.random.randint(365, 365 * 5))).date() for _ in range(num_rows)],
    "last_service_date": [(datetime.now() - timedelta(days=np.random.randint(0, 365))).date() for _ in range(num_rows)],
    "service_type": np.random.choice(["Oil Change", "Tire Rotation", "Major Repair"], num_rows),
    "cost_of_service": np.random.uniform(50, 1500, num_rows).round(2),
    "next_service_date": [(datetime.now() + timedelta(days=np.random.randint(30, 365))).date() for _ in range(num_rows)],
    "operational_status": np.random.choice(["Operational", "Maintenance", "Decommissioned"], num_rows),
    "fuel_consum_start": np.random.uniform(5, 20, num_rows).round(2),
    "fuel_consum_end": np.random.uniform(5, 20, num_rows).round(2),
    "driver_assigned": np.random.choice(["Driver A", "Driver B", "Driver C", "Driver D"], num_rows),
    "length_mean": np.random.uniform(2, 12, num_rows).round(2),
    "length_total": np.random.uniform(100, 1000, num_rows).round(2),
    "fuel_pos": np.random.uniform(1, 10, num_rows).round(2),
    "fuel_neg": np.random.uniform(1, 10, num_rows).round(2)
}

# Handle dependent data generation for 'current_mileage'
data["current_mileage"] = [initial + np.random.randint(1000, 50000) for initial in data["initial_mileage"]]

# Convert the data dictionary to DataFrame
df = pd.DataFrame(data)

# Write to Excel
df.to_excel("vehicles_data.xlsx", index=False)

df.head()  # Display the first few rows to check
