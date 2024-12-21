from flask import Flask, render_template, request
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage
from pinecone import Pinecone
import pandas as pd
import time
import os  
from datasets import load_dataset
from tqdm.auto import tqdm

app = Flask(__name__, static_url_path='/static')

# Initialize OpenAIEmbeddings
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",
                               openai_api_key="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")

# Initialize Pinecone
pc = Pinecone(api_key="c8b8503f-7230-4bab-a4fa-57f638bfc7c6")
index_name = 'testcone'

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        index_name,
        dimension=1536,
        metric='euclidean',
        spec=PodSpec(environment="gcp-starter")
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)

# Define the list of dataset paths
dataset_paths = [
   "C:\\Users\\mirut\\OneDrive\\Desktop\\YEAR 2\\INT2\\Miruthula's Datasets\\Miruthula's Datasets\\Amtrak_Bus_Stations.csv",
    "C:\\Users\\mirut\\OneDrive\\Desktop\\YEAR 2\\INT2\\Miruthula's Datasets\\Miruthula's Datasets\\Automated_Weather_Observation_System.csv"
] 

# Read each dataset into a DataFrame
datasets = [pd.read_csv(path) for path in dataset_paths]

batch_size = 100  # Specify the size of each batch
batches = [df[i:i+batch_size] for df in datasets for i in range(0, len(df), batch_size)]
    

# Process each dataset
for dataset_path in dataset_paths:
    # Load the dataset
    dataset = load_dataset("csv", data_files=[dataset_path], split="train")

    # Convert dataset to pandas DataFrame
    data = dataset.to_pandas()    

    # Process and embed text from each dataset
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['OBJECTID']}" for _, x in batch.iterrows()]  
        texts = [str(x['X']) for _, x in batch.iterrows() if isinstance(x['X'], str)]  # Filter out non-string values
        embeds = embed_model.embed_documents(texts)
        metadata = [{'text': str(x['X']), 'source': x['LATITUDE'], 'title': x['COUNTY']} for _, x in
                    batch.iterrows() if isinstance(x['X'], str)]  # Filter out non-string values and use str() to ensure all are strings
        
        if embeds:  # Check if embeds list is not empty
            index.upsert(vectors=zip(ids, embeds, metadata))



# Initialize PineconeVectorStore
text_field = "text"
vectorstore = PineconeVectorStore(index, embed_model, text_field)

# Initialize ChatOpenAI
chat = ChatOpenAI(openai_api_key="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", model='gpt-3.5-turbo')

conversation_history = []


@app.route('/', methods=['GET', 'POST'])
def index():
    global conversation_history
    if request.method == 'POST':
        # Check if 'query' key exists in the form data
        query = request.form.get('query')
        if query:  # If 'query' exists
            conversation_history.append({'type': 'user', 'content': query})

            # Retrieve response from ChatOpenAI
            response = chat.invoke([HumanMessage(content=query)]).content
            conversation_history.append({'type': 'response', 'content': response})

            # Retrieve response from Pinecone
            pinecone_response = vectorstore.similarity_search(query, k=1)
            if pinecone_response:
                pinecone_result = pinecone_response[0].metadata
                if 'text' in pinecone_result:
                    pinecone_response_text = f"Source: {pinecone_result['source']}, Location: {pinecone_result['text']}"
                    conversation_history.append({'type': 'pinecone_response', 'content': pinecone_response_text})
                else:
                    conversation_history.append(
                        {'type': 'pinecone_response', 'content': "Missing text information in Pinecone result"})
            else:
                conversation_history.append(
                    {'type': 'pinecone_response', 'content': "I don't have enough context to answer this query"})

    return render_template('index.html', messages=conversation_history)


if __name__ == '__main__':
    app.run(debug=True)
