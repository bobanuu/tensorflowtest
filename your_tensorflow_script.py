import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Step 1: Load the Data
data = pd.read_csv("/Users/benjones/Desktop/testmodel/Tensor Flow Testing - Copy of alltime_tensorflowtest.csv")

# Add a rating column (binary for simplicity: 1 if bought)
data['Rating'] = 1

# Step 2: Preprocess the Data
order_ids = data['Order ID'].astype('category').cat.codes.values
product_ids = data['SKU'].astype('category').cat.codes.values
ratings = data['Rating'].values
order_mapping = dict(enumerate(data['Order ID'].astype('category').cat.categories))
product_mapping = dict(enumerate(data['SKU'].astype('category').cat.categories))

# Step 3: Build a Neural Network Model for Recommendations
EMBEDDING_SIZE = 50
NUM_ORDERS = len(data['Order ID'].unique())
NUM_PRODUCTS = len(data['SKU'].unique())

input_orders = layers.Input(shape=(1,))
input_products = layers.Input(shape=(1,))
embedding_orders = layers.Embedding(NUM_ORDERS + 1, EMBEDDING_SIZE)(input_orders)
embedding_products = layers.Embedding(NUM_PRODUCTS + 1, EMBEDDING_SIZE)(input_products)

dot_product = layers.Dot(axes=2)([embedding_orders, embedding_products])
flatten = layers.Flatten()(dot_product)
output = layers.Dense(1)(flatten)

model = tf.keras.Model(inputs=[input_orders, input_products], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit([order_ids, product_ids], ratings, epochs=10, batch_size=64, validation_split=0.1)

# Step 5: Generate Recommendations for a specific Order ID
def get_recommendations_for_order_id(order_id_str):
    # Convert the order_id_str to its encoded value
    order_id_encoded = data['Order ID'].astype('category').cat.categories.get_loc(order_id_str)
    
    # Extract SKUs already in the order
    existing_products = set(data[data['Order ID'] == order_id_str]['SKU'].values)
    
    order_vector = np.repeat(order_id_encoded, NUM_PRODUCTS)
    product_vector = np.arange(NUM_PRODUCTS)
    
    predictions = model.predict([order_vector, product_vector])
    recommended_products_indices = (-predictions).argsort(axis=0).flatten()
    
    sku_to_description = dict(zip(data['SKU'], data['Description']))
    
    recommended_products = [sku_to_description[product_mapping[idx]] for idx in recommended_products_indices if product_mapping[idx] not in existing_products]

    print(f"Recommendations for order: {order_id_str}")
    return recommended_products

# Call the recommendation function for a specific Order ID:
specific_order_id = "UberEats-181d32c1-a0c5-40bc-b0f5-f55d4efb863f"
recommendations = get_recommendations_for_order_id(specific_order_id)
print(recommendations[:3])  # Print the top 3 recommended product descriptions for the specified order
