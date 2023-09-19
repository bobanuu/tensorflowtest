from flask import Flask, render_template, request
import requests
from io import StringIO
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Function to Load Data from Google Drive
def load_data_from_gdrive():
    GOOGLE_DRIVE_FILE_ID = '1eK_XNPECPgsJQTm1z6RtdstvV7fYr_Dx'
    GOOGLE_DRIVE_DOWNLOAD_URL = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'
    
    response = requests.get(GOOGLE_DRIVE_DOWNLOAD_URL)
    response.raise_for_status()

    csv_content = StringIO(response.text)
    return pd.read_csv(csv_content)

# Step 1: Load the Data
data = load_data_from_gdrive()
data['Rating'] = 1

# Step 2: Preprocess the Data
order_ids = data['Order ID'].astype('category').cat.codes.values
product_ids = data['SKU'].astype('category').cat.codes.values
ratings = data['Rating'].values

NUM_ORDERS = len(data['Order ID'].unique())
NUM_PRODUCTS = len(data['SKU'].unique())

order_id_to_encoded = dict(zip(data['Order ID'].unique(), range(NUM_ORDERS)))
product_mapping = dict(enumerate(data['SKU'].astype('category').cat.categories))

# Step 3: Build a Neural Network Model for Recommendations
EMBEDDING_SIZE = 20

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
model.fit([order_ids, product_ids], ratings, epochs=5, batch_size=64, validation_split=0.1)

# Step 5: Generate Recommendations for a specific Order ID
def get_recommendations(order_id_str):
    order_id_encoded = order_id_to_encoded[order_id_str]
    existing_products = set(data[data['Order ID'] == order_id_str]['SKU'].values)
    
    order_vector = np.repeat(order_id_encoded, NUM_PRODUCTS)
    product_vector = np.arange(NUM_PRODUCTS)
    
    predictions = model.predict([order_vector, product_vector])
    recommended_products_indices = (-predictions).argsort(axis=0).flatten()
    
    sku_to_description = dict(zip(data['SKU'], data['Description']))
    recommended_products = [sku_to_description[product_mapping[idx]] for idx in recommended_products_indices if product_mapping[idx] not in existing_products]

    return recommended_products

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        order_id = request.form['order_id']
        if order_id in order_id_to_encoded:
            recommendations = get_recommendations(order_id)[:3]
            recommendations = [str(item) for item in recommendations]
            if not recommendations:
                recommendations = ["Oreos", "Kitkat", "CocaCola"]
        else:
            recommendations = ["Order ID not found in the dataset!"]

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
