import time
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import google.generativeai as genai
from cvm_rag import preprocessing
import os

API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
RDS_CVM_RAG_DB_PW = os.getenv("RDS_CVM_RAG_DB_PW")
RDS_DB_URL = os.getenv("RDS_DB_URL")

genai.configure(api_key=API_KEY_GEMINI)

def slow_echo(message, history):
    """
    Generates a response to the input message using a generative AI model.
    
    Args:
        message (str): The input message.
        history (list): The conversation history.
    
    Yields:
        str: The generated response text.
    """
    documents = query_database(message)
    document_text = str(documents)

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""Assuma que você é um analista financeiro.\nVocê vai receber informações sobre o formulário de referência de uma empresa entre a tag <data> e vai responder as perguntas relacionadas a empresa. 
    'doc_text' se refere ao texto do documento e 'denom_cia' a empresa associada ao documento
    Responda de forma resumida e esteja preparado para fazer cálculos. 

    <data>
    {document_text}
    </data>
    A resposta para a mensagem: {message} é:
    """

    response = model.generate_content(prompt)

    for i in range(len(response.text)):
        time.sleep(0.001)
        yield response.text[: i + 1]

def query_database(input_text):
    """
    Queries the database for documents related to the input text.
    
    Args:
        input_text (str): The input text to query.
    
    Returns:
        list: A list of documents related to the input text.
    """
    conn_params = {
        'dbname': 'db_name',
        'user': 'postgres',
        'password': RDS_CVM_RAG_DB_PW,
        'host': RDS_DB_URL,
        'port': '5432'
    }

    try:
        with psycopg2.connect(**conn_params) as connection:
            register_vector(connection)
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                input_text_embedding = preprocessing.generate_embeddings(input_text)
                query_embedding_str = ','.join(map(str, input_text_embedding))

                cursor.execute(f"""SELECT d.doc_text, fca.denom_cia
                                   FROM documents d LEFT JOIN fre_cia_aberta fca on d.id_doc = fca.id_doc
                                   ORDER BY (d.doc_embeddings <=> ARRAY[{query_embedding_str}]::vector) 
                                   LIMIT 10;""")
                documents = cursor.fetchall()
    except Exception as e:
        print(f"An error occurred: {e}")
        documents = []

    return documents

app = gr.ChatInterface(slow_echo, type="messages", css="footer {visibility: hidden}", title="CVM RAG")

if __name__ == "__main__":
    app.launch(share=True)