import pandas as pd
import pickle

# Load the dataset
file_path = "traffic_dataset.csv"  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Define standard encoding order
label_encodings = {
    "Day_of_Week": {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6},
    "Month": {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6, 'August': 7, 
              'September': 8, 'October': 9, 'November': 10, 'December': 11},
    "Season": {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
}

# Apply encoding
df["Day_of_Week"] = df["Day_of_Week"].map(label_encodings["Day_of_Week"])
df["Month"] = df["Month"].map(label_encodings["Month"])
df["Season"] = df["Season"].map(label_encodings["Season"])

# Save the encoded dataset
encoded_file_path = "encoded_dataset.csv"
df.to_csv(encoded_file_path, index=False)
print(f"Encoded dataset saved as {encoded_file_path}")

# Save label encodings as a pickle file
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encodings, f)

print("Label encoder saved as label_encoder.pkl")


#################################################################
## Decoder Code Snippet
#################################################################
# import pandas as pd
# import pickle

# # Load the encoded dataset
# encoded_file_path = "encoded_dataset.csv"  # Replace with your actual file path
# df = pd.read_csv(encoded_file_path)

# # Load the label encoder mappings
# with open("label_encoder.pkl", "rb") as f:
#     label_encodings = pickle.load(f)

# # Reverse the encoding mappings
# inverse_label_encodings = {
#     key: {v: k for k, v in value.items()} for key, value in label_encodings.items()
# }

# # Apply decoding
# df["Day_of_Week"] = df["Day_of_Week"].map(inverse_label_encodings["Day_of_Week"])
# df["Month"] = df["Month"].map(inverse_label_encodings["Month"])
# df["Season"] = df["Season"].map(inverse_label_encodings["Season"])

# # Save the decoded dataset
# decoded_file_path = "decoded_dataset.csv"
# df.to_csv(decoded_file_path, index=False)
# print(f"Decoded dataset saved as {decoded_file_path}")
