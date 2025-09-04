---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code id="00QDUiCy7CiK"}
``` python
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="3NxCxlyC7Inj" outputId="62e4af2e-94ce-4d1c-ed01-4a460f69b981"}
``` python
# --- 1. Data Fetching ---
# Fetch product data from the Fake Store API.
try:
    print("Fetching data from Fake Store API...")
    response = requests.get('https://fakestoreapi.com/products')
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    products_data = response.json()
    print(f"Successfully fetched {len(products_data)} products.")
except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
    products_data = []

# Convert the JSON data to a pandas DataFrame for easier manipulation.
if products_data:
    products_df = pd.DataFrame(products_data)
else:
    print("No data fetched, creating an empty DataFrame.")
    products_df = pd.DataFrame()
```

::: {.output .stream .stdout}
    Fetching data from Fake Store API...
    Successfully fetched 20 products.
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="vRMcj8scJdld" outputId="028c5834-3bdb-4115-f069-3d29a689d4e4"}
``` python
# Basic dataset checks
if 'products_df' in locals():
    print("DataFrame loaded successfully ✅\n")
    print("Shape (rows, cols):", products_df.shape)
    print("\nColumn list:", products_df.columns.tolist())
    print("\nMissing values per column:\n", products_df.isnull().sum())
    print("\nData types:\n", products_df.dtypes)
    print("\nSample rows:\n", products_df.head(8))
else:
    print("products_df not found in memory ❌")
```

::: {.output .stream .stdout}
    DataFrame loaded successfully ✅

    Shape (rows, cols): (20, 7)

    Column list: ['id', 'title', 'price', 'description', 'category', 'image', 'rating']

    Missing values per column:
     id             0
    title          0
    price          0
    description    0
    category       0
    image          0
    rating         0
    dtype: int64

    Data types:
     id               int64
    title           object
    price          float64
    description     object
    category        object
    image           object
    rating          object
    dtype: object

    Sample rows:
        id                                              title   price  \
    0   1  Fjallraven - Foldsack No. 1 Backpack, Fits 15 ...  109.95   
    1   2             Mens Casual Premium Slim Fit T-Shirts    22.30   
    2   3                                 Mens Cotton Jacket   55.99   
    3   4                               Mens Casual Slim Fit   15.99   
    4   5  John Hardy Women's Legends Naga Gold & Silver ...  695.00   
    5   6                       Solid Gold Petite Micropave   168.00   
    6   7                         White Gold Plated Princess    9.99   
    7   8  Pierced Owl Rose Gold Plated Stainless Steel D...   10.99   

                                             description        category  \
    0  Your perfect pack for everyday use and walks i...  men's clothing   
    1  Slim-fitting style, contrast raglan long sleev...  men's clothing   
    2  great outerwear jackets for Spring/Autumn/Wint...  men's clothing   
    3  The color could be slightly different between ...  men's clothing   
    4  From our Legends Collection, the Naga was insp...        jewelery   
    5  Satisfaction Guaranteed. Return or exchange an...        jewelery   
    6  Classic Created Wedding Engagement Solitaire D...        jewelery   
    7  Rose Gold Plated Double Flared Tunnel Plug Ear...        jewelery   

                                                   image  \
    0  https://fakestoreapi.com/img/81fPKd-2AYL._AC_S...   
    1  https://fakestoreapi.com/img/71-3HjGNDUL._AC_S...   
    2  https://fakestoreapi.com/img/71li-ujtlUL._AC_U...   
    3  https://fakestoreapi.com/img/71YXzeOuslL._AC_U...   
    4  https://fakestoreapi.com/img/71pWzhdJNwL._AC_U...   
    5  https://fakestoreapi.com/img/61sbMiUnoGL._AC_U...   
    6  https://fakestoreapi.com/img/71YAIFU48IL._AC_U...   
    7  https://fakestoreapi.com/img/51UDEzMJVpL._AC_U...   

                            rating  
    0  {'rate': 3.9, 'count': 120}  
    1  {'rate': 4.1, 'count': 259}  
    2  {'rate': 4.7, 'count': 500}  
    3  {'rate': 2.1, 'count': 430}  
    4  {'rate': 4.6, 'count': 400}  
    5   {'rate': 3.9, 'count': 70}  
    6    {'rate': 3, 'count': 400}  
    7  {'rate': 1.9, 'count': 100}  
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="tnHx4lX07PN5" outputId="c73596f0-8416-483b-cf9e-e39384bd9acc"}
``` python
# --- 2. Data Preprocessing ---
if not products_df.empty:
    # Fill any missing values in key text fields to avoid errors.
    products_df['title'] = products_df['title'].fillna('')
    products_df['description'] = products_df['description'].fillna('')
    products_df['category'] = products_df['category'].fillna('')

    # Combine the relevant text fields into a single string for each product.
    products_df['soup'] = products_df['title'] + ' ' + products_df['description'] + ' ' + products_df['category']
    print("\nSample of the combined text 'soup' for a product:")
    print(products_df['soup'].head(1).values[0])
```

::: {.output .stream .stdout}

    Sample of the combined text 'soup' for a product:
    Fjallraven - Foldsack No. 1 Backpack, Fits 15 Laptops Your perfect pack for everyday use and walks in the forest. Stash your laptop (up to 15 inches) in the padded sleeve, your everyday men's clothing
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1926}" id="lADzEEKHIB9X" outputId="ff39c35f-7848-42e4-8022-8800abe273c0"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns

# Category distribution
if 'category' in products_df.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(data=products_df, x='category', order=products_df['category'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Category Distribution")
    plt.show()

# Price distribution
if 'price' in products_df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(products_df['price'].dropna(), bins=30, kde=True)
    plt.title("Price Distribution")
    plt.show()

# Rating distribution
if 'rating' in products_df.columns and not products_df['rating'].empty:
    # Extract the 'rate' from the rating dictionary
    products_df['rating_rate'] = products_df['rating'].apply(lambda x: x['rate'] if isinstance(x, dict) and 'rate' in x else None)
    plt.figure(figsize=(8,5))
    sns.histplot(products_df['rating_rate'].dropna(), bins=20, kde=False)
    plt.title("Rating Distribution")
    plt.show()

# Description length analysis
if 'description' in products_df.columns:
    products_df['desc_length'] = products_df['description'].astype(str).apply(len)
    plt.figure(figsize=(8,5))
    sns.histplot(products_df['desc_length'], bins=30, kde=True)
    plt.title("Description Length Distribution")
    plt.show()

    print("\nTop 10 longest descriptions:")
    display(products_df[['title','desc_length','description']].sort_values(by='desc_length', ascending=False).head(10))
```

::: {.output .display_data}
![](ee6ff17a8dc285671db18353e5e39d613e4d2db3.png)
:::

::: {.output .display_data}
![](dda68ba2e5a20a6a4a9d1ae6e131b9e0354a0bc3.png)
:::

::: {.output .display_data}
![](1fe8868349614ec6c63d70b2231c67710b8210da.png)
:::

::: {.output .display_data}
![](61ad9ea23d7c7e9e2ff41434ca01847cfd873db6.png)
:::

::: {.output .stream .stdout}

    Top 10 longest descriptions:
:::

::: {.output .display_data}
``` json
{"summary":"{\n  \"name\": \"    display(products_df[['title','desc_length','description']]\",\n  \"rows\": 10,\n  \"fields\": [\n    {\n      \"column\": \"title\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 10,\n        \"samples\": [\n          \"Mens Cotton Jacket\",\n          \"SanDisk SSD PLUS 1TB Internal SSD - SATA III 6 Gb/s\",\n          \"Opna Women's Short Sleeve Moisture\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"desc_length\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 139,\n        \"min\": 307,\n        \"max\": 772,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          336,\n          495,\n          347\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"description\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 10,\n        \"samples\": [\n          \"great outerwear jackets for Spring/Autumn/Winter, suitable for many occasions, such as working, hiking, camping, mountain/rock climbing, cycling, traveling or other outdoors. Good gift choice for you or your family member. A warm hearted love to Father, husband or son in this thanksgiving or Christmas Day.\",\n          \"Easy upgrade for faster boot up, shutdown, application load and response (As compared to 5400 RPM SATA 2.5\\u201d hard drive; Based on published specifications and internal benchmarking tests using PCMark vantage scores) Boosts burst write performance, making it ideal for typical PC workloads The perfect balance of performance and reliability Read/write speeds of up to 535MB/s/450MB/s (Based on internal testing; Performance may vary depending upon drive capacity, host device, OS and application.)\",\n          \"100% Polyester, Machine wash, 100% cationic polyester interlock, Machine Wash & Pre Shrunk for a Great Fit, Lightweight, roomy and highly breathable with moisture wicking fabric which helps to keep moisture away, Soft Lightweight Fabric with comfortable V-neck collar and a slimmer fit, delivers a sleek, more feminine silhouette and Added Comfort\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":449}" id="_4VyGcjE7mat" outputId="e6f32237-03c5-44ed-c090-18fb92171fb1"}
``` python
# --- 3. Neural Network Feature Extraction ---
# We will now use a neural network to convert the text 'soup' into dense embeddings.

# Step 3.1: Tokenization
# Convert the text into a sequence of integers.
print("\nTokenizing text data...")
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(products_df['soup'])
sequences = tokenizer.texts_to_sequences(products_df['soup'])
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

# Step 3.2: Padding
# Ensure all sequences have the same length by padding them.
print("Padding sequences...")
max_length = 100 # We can choose an appropriate max length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
print("Shape of padded sequences:", padded_sequences.shape)

# Step 3.3: Build the Embedding Model
# We define a simple model to learn embeddings.
# This model is not trained in a traditional sense for this task;
# we use it to transform our text data into meaningful vectors.
vocab_size = len(word_index) + 1
embedding_dim = 16 # Dimensionality of the embedding vector

input_layer = Input(shape=(max_length,))
# The Embedding layer turns positive integers (indexes) into dense vectors of fixed size.
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
# GlobalAveragePooling1D averages the embeddings for all words in a product's description
# to get a single vector representation for the entire product.
pooling_layer = GlobalAveragePooling1D()(embedding_layer)

# Create a model that will output the embeddings
embedding_model = Model(inputs=input_layer, outputs=pooling_layer)
embedding_model.summary()

# Step 3.4: Generate Product Embeddings
# Use the model to predict (generate) the embeddings for our padded sequences.
print("\nGenerating product embeddings using the neural network...")
product_embeddings = embedding_model.predict(padded_sequences)
print("Shape of product embeddings matrix:", product_embeddings.shape)


# --- 4. Model Training (Cosine Similarity on Embeddings) ---
# Compute the cosine similarity between all pairs of product embeddings.
print("\nCalculating cosine similarity matrix on embeddings...")
cosine_sim = cosine_similarity(product_embeddings, product_embeddings)
print("Shape of the cosine similarity matrix:", cosine_sim.shape)

# Create a mapping from product titles to their index in the DataFrame.
indices = pd.Series(products_df.index, index=products_df['title']).drop_duplicates()
```

::: {.output .stream .stdout}

    Tokenizing text data...
    Found 593 unique tokens.
    Padding sequences...
    Shape of padded sequences: (20, 100)
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ embedding (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │         <span style="color: #00af00; text-decoration-color: #00af00">9,504</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)        │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,504</span> (37.12 KB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,504</span> (37.12 KB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>
```
:::

::: {.output .stream .stdout}

    Generating product embeddings using the neural network...
    1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 148ms/step
    Shape of product embeddings matrix: (20, 16)

    Calculating cosine similarity matrix on embeddings...
    Shape of the cosine similarity matrix: (20, 20)
:::
:::

::: {.cell .code id="keSPAjxV7uhu"}
``` python
# --- 5. Recommendation Function ---
def get_recommendations(title, cosine_sim=cosine_sim, data=products_df):

    if title not in indices.index:
        return f"Product with title '{title}' not found."

    # Get the index of the product that matches the title.
    idx = indices[title]

    # Get the pairwise similarity scores of all products with that product.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores in descending order.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products. Skip the first one.
    sim_scores = sim_scores[1:11]

    # Get the product indices from the similarity scores.
    product_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar products.
    return data['title'].iloc[product_indices]
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="V6pdx_iW6lQ1" outputId="0b15b131-d3ec-4e4c-a8a2-b951c1d4111f"}
``` python
# --- 6. Example Usage ---
print("\n--- Recommendation Example ---")
if not products_df.empty:
    example_product_title = products_df['title'][0]
    print(f"Recommendations for: '{example_product_title}'\n")
    recommendations = get_recommendations(example_product_title)
    print(recommendations)

    print("\n" + "="*30 + "\n")
    example_product_title_2 = products_df['title'][5]
    print(f"Recommendations for: '{example_product_title_2}'\n")
    recommendations_2 = get_recommendations(example_product_title_2)
    print(recommendations_2)
else:
    print("Cannot generate recommendations because no product data is available.")

# This `else` block was causing the IndentationError
# else:
#     print("\nScript finished. Could not proceed with model training due to data fetching issues.")
```

::: {.output .stream .stdout}

    --- Recommendation Example ---
    Recommendations for: 'Fjallraven - Foldsack No. 1 Backpack, Fits 15 Laptops'

    6                            White Gold Plated Princess
    11    WD 4TB Gaming Drive Works with Playstation 4 P...
    3                                  Mens Casual Slim Fit
    7     Pierced Owl Rose Gold Plated Stainless Steel D...
    4     John Hardy Women's Legends Naga Gold & Silver ...
    2                                    Mens Cotton Jacket
    19           DANVOUY Womens T Shirt Casual Cotton Short
    1                Mens Casual Premium Slim Fit T-Shirts 
    5                          Solid Gold Petite Micropave 
    17          MBJ Women's Solid Short Sleeve Boat Neck V 
    Name: title, dtype: object

    ==============================

    Recommendations for: 'Solid Gold Petite Micropave '

    7     Pierced Owl Rose Gold Plated Stainless Steel D...
    6                            White Gold Plated Princess
    11    WD 4TB Gaming Drive Works with Playstation 4 P...
    2                                    Mens Cotton Jacket
    4     John Hardy Women's Legends Naga Gold & Silver ...
    3                                  Mens Casual Slim Fit
    17          MBJ Women's Solid Short Sleeve Boat Neck V 
    0     Fjallraven - Foldsack No. 1 Backpack, Fits 15 ...
    8     WD 2TB Elements Portable External Hard Drive -...
    19           DANVOUY Womens T Shirt Casual Cotton Short
    Name: title, dtype: object
:::
:::

::: {.cell .code id="8fS5yyWNU9b6"}
``` python
```
:::
