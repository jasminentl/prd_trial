import os
import pandas as pd
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from neo4j import GraphDatabase


# Load environment variables
load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)

# Initialize Neo4j Driver
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

TARGET_DATABASE = os.getenv("NEO4J_DATABASE")


def prepare_onet_data(file_path="content_model_reference.csv") -> pd.DataFrame:
    """Filter O*NET CSV for Work Styles, Essential, and Transferable Skills."""
    df = pd.read_csv(file_path)

    # Filter by Element ID prefixes for soft skills & behavioral work styles
    mask = df["Element ID"].str.startswith(("1.A.1", "2.A", "2.B", "2.C"))
    filtered_df = df[mask].copy()

    # Filter out high-level category titles (keep specific leaf elements)
    filtered_df = filtered_df[filtered_df["Element ID"].str.len() > 3]

    print(f"Dataset prepared: {len(filtered_df)} O*NET behavioral elements found.")
    return filtered_df


def setup_neo4j_vector_index(session):
    """Create a native vector index in Neo4j if it does not exist."""
    create_index_query = """
    CREATE VECTOR INDEX onet_vector_index IF NOT EXISTS
    FOR (n:ONetElement) ON (n.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }
    };
    """
    session.run(create_index_query)
    print(f"Vector index 'onet_vector_index' configured on database '{TARGET_DATABASE}'.")


def ingest_data_to_neo4j(df: pd.DataFrame):
    """Generate embeddings and write nodes into the target Neo4j database."""
    with neo4j_driver.session(database=TARGET_DATABASE) as session:
        setup_neo4j_vector_index(session)

        for _, row in df.iterrows():
            element_id = row["Element ID"]
            name = row["Element Name"]
            description = row["Description"]

            # Combine Name and Description for richer embedding representation
            text_to_embed = f"{name}: {description}"
            embedding = embeddings_model.embed_query(text_to_embed)

            # Upsert query targeting the isolated graph database
            upsert_query = """
            MERGE (e:ONetElement {element_id: $element_id})
            SET e.name = $name,
                e.description = $description,
                e.embedding = $embedding
            """
            session.run(
                upsert_query,
                element_id=element_id,
                name=name,
                description=description,
                embedding=embedding,
            )
            print(f"Ingested: [{element_id}] {name}")


if __name__ == "__main__":
    try:
        data = prepare_onet_data("Data/content_model_reference.csv")
        ingest_data_to_neo4j(data)
        print("\nData ingestion and vector indexing completed successfully!")
    finally:
        neo4j_driver.close()