from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the index name
index_name = "contract-embeddings"

# Function to generate embeddings
def generate_embedding(text):
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return result.data[0].values

# Function to search for similar vectors
def search_similar_vectors(query_vector, top_k=5):
    index = pc.Index(index_name)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results

# Function to rerank results
def rerank_results(query, documents):
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=True,
        parameters={"truncate": "END"}
    )
    return result.data

# Function to format vector details
def format_vector_details(vector, rank, score):
    return {
        "rank": rank,
        "record_id": vector.id,
        "score": round(score, 4),
        "title_description": vector.metadata.get('title_description', 'N/A'),
        "metadata": {
            "additional_classifications": vector.metadata.get('additional_classifications', 'N/A'),
            "buyer_city": vector.metadata.get('buyer_city', 'N/A'),
            "buyer_country": vector.metadata.get('buyer_country', 'N/A'),
            "buyer_email": vector.metadata.get('buyer_email', 'N/A'),
            "buyer_name": vector.metadata.get('buyer_name', 'N/A'),
            "buyer_website": vector.metadata.get('buyer_website', 'N/A'),
            "currency": vector.metadata.get('currency', 'N/A'),
            "deadline_date": vector.metadata.get('deadline_date', 'N/A'),
            "estimated_value": vector.metadata.get('estimated_value', 'N/A'),
            "main_classification": vector.metadata.get('main_classification', 'N/A'),
            "main_nature": vector.metadata.get('main_nature', 'N/A'),
            "procedure_type": vector.metadata.get('procedure_type', 'N/A'),
            "text": vector.metadata.get('text', 'N/A')[:100] + "..."  # Truncate text to first 100 characters
        }
    }

def lambda_handler(event, context):
    try:

        # Parse input text from the event's body (API Gateway sends the body as a string)
        body = json.loads(event['body'])
        input_text = body.get('text', '')
        
        # Generate embedding for the input
        query_vector = generate_embedding(input_text)

        # Search for similar vectors in the contract-embeddings index
        search_results = search_similar_vectors(query_vector)

        # Prepare documents for reranking
        documents = [
            {"id": match.id, "text": match.metadata.get("title_description", "")}
            for match in search_results.matches
        ]

        # Rerank the results
        reranked_results = rerank_results(input_text, documents)

        # Format the final results with detailed information
        formatted_results = [
            {"record_id": result.document['id']}
            for result in reranked_results
        ]

        # Return the results
        return {
            'statusCode': 200,
            'body': json.dumps(formatted_results),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # Allow CORS for your NextJS app
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }

# This is only for local testing
if __name__ == "__main__":
    test_event = {
        'body': "I am looking for a power tools / industrial equipment contract, as I have a company that sells power tools and industrial equipment, based in the UK and Lithuania."
    }
    print(json.dumps(lambda_handler(test_event, None), indent=2))
