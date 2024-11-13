from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the index name
index_name = "lot-parse"

# Function to generate embeddings
def generate_embedding(text):
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return result.data[0].values

# Function to search for similar vectors
def search_similar_vectors(query_vector, top_k=20):
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

def search_contracts(input_text):
    try:
        # Generate embedding for the input
        query_vector = generate_embedding(input_text)

        # Search for similar vectors
        search_results = search_similar_vectors(query_vector)

        # Prepare documents for reranking
        documents = [
            {"id": match.id, "text": match.metadata.get("title_description", "")}
            for match in search_results.matches
        ]

        # Rerank the results
        reranked_results = rerank_results(input_text, documents)

        # Format the final results
        formatted_results = []
        for result in reranked_results:
            formatted_result = {
                "record_id": result.document['id'],
                "score": result.score,
                "metadata": {
                    "title_description": result.document.get("text", "N/A"),
                }
            }
            formatted_results.append(formatted_result)

        return formatted_results
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

if __name__ == "__main__":
    # Your company description
    company_description = """
    Holden Plant Rentals is Ireland's premier provider of specialized machinery and vehicle rental services, as well as comprehensive fleet management solutions. Headquartered in Mullinavat, County Kilkenny, the company delivers a diverse suite of products and services designed to support industries ranging from construction to public service.

Core Services and Offerings:
Machinery Rental With an extensive fleet of over 4,000 premium machines, Holden Plant Rentals provides:

Construction Equipment: Reliable machinery for building and infrastructure projects.
Specialized Public Works Machinery: Tailored machines for large-scale public sector projects.
Tractors: Versatile tractors suitable for agricultural, industrial, and airport operations.
Vehicle Rental Holden Plant Rentals offers a robust vehicle rental service, specializing in:

Long-Term Rentals: Custom contracts for corporate clients and government entities, ensuring flexibility and convenience.
Specialized Vehicles: Purpose-built vehicles to meet specific requirements across various industries.
Fleet Management Services The company’s fleet management division offers end-to-end solutions, including:

Vehicle Licensing and Administration
Tax Renewal and Compliance
NCT (National Car Test) Coordination
Comprehensive Servicing and Maintenance
Overhead Management: Including tolls, parking fines, and penalty points management.
24-Hour Support: Ensuring continuous assistance and peace of mind.
    """

    print("Searching for relevant contracts...")
    results = search_contracts(company_description)
    
    print("\nTop matching contracts:")
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. Score: {result['score']:.4f}")
        print(f"Record ID: {result['record_id']}")
        print(f"Title: {result['metadata']['title_description']}")