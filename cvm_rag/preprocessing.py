import pandas as pd
import polars as pl
from datetime import datetime
import zipfile
from bs4 import BeautifulSoup
import base64
import requests
import shutil
import os
from pypdf import PdfMerger
from pathlib import Path
import pymupdf  # PyMuPDF
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql
from typing import List, Dict
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import concurrent.futures

def download_file(url: str, output_file: str):
    """
    Downloads a file from a given URL and saves it to the specified output path.

    Args:
        url (str): The URL of the file to download.
        output_file (str): The path where the downloaded file will be saved.
    """
    try:
        if os.path.exists(output_file):
            print(f"File {output_file} already exists")
            return

        print(f"Starting to download {url}")
        with requests.get(url, stream=True, headers={'Accept-Encoding': 'identity'}) as response:
            response.raise_for_status()
            with open(output_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"Download completed: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_docs(df: pd.DataFrame):
    """
    Downloads documents listed in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing document IDs and URLs.
    """
    output_dir = "downloads/"
    os.makedirs(output_dir, exist_ok=True)

    def process_row(row):
        filename, url = row
        download_file(url, os.path.join(output_dir, str(filename)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_row, df[["ID_DOC", "LINK_DOC"]].iter_rows())

def extract_file(file: str) -> str:
    """
    Extracts a specific file from a ZIP archive and returns its content.

    Args:
        file (str): Path to the ZIP file.

    Returns:
        str: Content of the extracted file.
    """
    filename = os.path.basename(file)
    with zipfile.ZipFile(file, 'r') as zip_file:
        fre_file_to_read = [item for item in zip_file.namelist() if "fre" in item.lower()][0]
        with zip_file.open(fre_file_to_read) as source_file:
            with open(f"extracted_files/{filename}.xml", 'wb') as dest_file:
                dest_file.write(source_file.read())

    with open(f"extracted_files/{filename}.xml", 'r', encoding="latin1") as f:
        data = f.read()

    return data

def get_list_files(path: str) -> List[str]:
    """
    Returns a list of file paths in the specified directory.

    Args:
        path (str): Directory path.

    Returns:
        List[str]: List of file paths.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def generate_fre_pdfs(file: str) -> None:
    """
    Generates PDFs from extracted XML data.

    Args:
        file (str): Path to the ZIP file containing XML data.
    """
    print(f"Processing {file}")
    try:
        data = extract_file(file)
    except Exception as e:
        print(f"Error extracting file: {file}, {e}")
        return

    bs_data = BeautifulSoup(data, "xml")
    all_pdfs = bs_data.find_all("ImagemObjetoArquivoPdf")
    print(len(all_pdfs))

    date = datetime.now().strftime("%Y%m%d")
    original_filename = os.path.basename(file)
    path = f"pdfs/{original_filename}"

    os.makedirs(path, exist_ok=True)

    for i, pdf in enumerate(all_pdfs):
        bytes_pdf = pdf.text.encode("ascii")
        pdf_code = base64.decodebytes(bytes_pdf)
        if not pdf_code:
            break

        with open(f"{path}/{date}_{i:04}.pdf", "ab") as f:
            f.write(pdf_code)

    lst = sorted(os.listdir(path))
    merger = PdfMerger()
    for pdf in lst:
        merger.append(f"{path}/{pdf}")

    merger.write(f"pdfs/{original_filename}.pdf")
    merger.close()
    shutil.rmtree(path)

def extract_text_from_pdf(file: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    with open(file, 'rb') as pdf_file:
        doc = pymupdf.open(pdf_file)
        extracted_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted_text += page.get_text() + "\n"
    return extracted_text

def generate_ner(text: str, model_name: str='Babelscape/wikineural-multilingual-ner'):
    """
    Generates Named Entity Recognition (NER) results for the given text.

    Args:
        text (str): Input text.
        model_name (str): Name of the NER model.

    Returns:
        List[Dict]: NER results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    return ner_pipeline(text)

def generate_text_metadata(text: str, model_name: str='microsoft/Phi-3-mini-4k-instruct'):
    """
    Generates metadata for the given text using a text generation model.

    Args:
        text (str): Input text.
        model_name (str): Name of the text generation model.

    Returns:
        str: Generated metadata.
    """
    metadata_pipeline = pipeline("text-generation", model=model_name)
    prompt = f"""Gere metadados para o texto abaixo em JSON que será processado em Python. Use uma estrutura similar a essa:
    {{
        "page_number": 1,
        "title": 'Formulario de Referencia da Empresa',
        "company": 'Company Name',
        "questions_this_excerpt_can_answer": '1. Question 1\n2. Question 2\n3. Question 3.'
    }}

    <data>
    {text}
    </data>
    Os metadados em JSON são:
    """
    response = metadata_pipeline(prompt, return_text=True)
    return response

def process_json_markdown(text: str):
    """
    Processes JSON data embedded in a Markdown string.

    Args:
        text (str): Input Markdown string.

    Returns:
        Dict: Parsed JSON data.
    """
    start = text.find('```json\n') + len('```json\n')
    end = text.rfind('\n```')
    if start != -1 and end != -1:
        json_string = text[start:end]
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
    else:
        print("No JSON block found.")

def generate_embeddings(text: str, model_name: str='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Generates embeddings for the given text.

    Args:
        text (str): Input text.
        model_name (str): Name of the embedding model.

    Returns:
        List[float]: Generated embeddings.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)

def split_text(text: str, chunk_size: int=1024, overlap: int=50, prefix: str='') -> List[str]:
    """
    Splits text into chunks of specified size with overlap.

    Args:
        text (str): Input text.
        chunk_size (int): Size of each chunk.
        overlap (int): Overlap between consecutive chunks.
        prefix (str): Prefix to add to each chunk.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0
    text = text.replace('\n', '')

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(prefix + text[start:end])
        start = end - overlap

    return chunks

def process_single_pdf(pdf_file: str, chunk_size: int = 1024, overlap: int = 200):
    """
    Processes a single PDF file: extracts text, splits into chunks, and generates embeddings.

    Args:
        pdf_file (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk.
        overlap (int): Overlap between consecutive chunks.

    Yields:
        Dict: A dictionary containing document ID, text, and embeddings for each chunk.
    """
    try:
        filename = os.path.basename(pdf_file)
        doc_id = os.path.splitext(filename)[0]
        text_input = extract_text_from_pdf(pdf_file)
        if not text_input:
            raise ValueError(f"No text extracted from {pdf_file}")

        chunks = split_text(text_input, chunk_size=chunk_size, overlap=overlap)
        chunks_embedding = generate_embeddings(chunks)

        if not chunks:
            raise ValueError(f"No chunks created from {pdf_file}")

        for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, chunks_embedding)):
            metadata = {
                "chunk_number": chunk_idx,
                "chunk_size": chunk_size,
                "filename": filename
            }
            yield {
                "doc_id": doc_id,
                "text": chunk,
                "embeddings": embedding,
                "metadata": json.dumps(metadata)
            }

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

def process_pdfs(pdf_files: List[str], use_threads: bool = True, chunk_size: int = 1024, overlap: int = 20):
    """
    Processes multiple PDF files in parallel and yields results lazily.

    Args:
        pdf_files (List[str]): List of PDF file paths.
        use_threads (bool): Use ThreadPoolExecutor if True, otherwise ProcessPoolExecutor.
        chunk_size (int): Size of each text chunk.
        overlap (int): Overlap between consecutive chunks.

    Yields:
        Dict: A dictionary containing document ID, text, and embeddings for each chunk.
    """
    executor_class = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
    try:
        with executor_class() as executor:
            for result in executor.map(lambda pdf: process_single_pdf(pdf, chunk_size, overlap), pdf_files):
                yield from result
    except Exception as e:
        print(f"Error processing PDFs: {e}")

def store_pdf_data_to_db(processed_data: List[Dict], conn_params: Dict):
    """
    Stores processed PDF data (doc_id, text, embeddings) into a PostgreSQL database.

    Args:
        processed_data (List[Dict]): List of dictionaries containing doc_id, text, and embeddings.
        conn_params (Dict): Database connection parameters.
    """
    connection = psycopg2.connect(**conn_params)
    register_vector(connection)
    cursor = connection.cursor()

    values = [(entry["doc_id"], entry["text"], entry["embeddings"]) for entry in processed_data]
    insert_query = sql.SQL("INSERT INTO documents (ID_DOC, DOC_TEXT, DOC_EMBEDDINGS) VALUES %s")

    try:
        execute_values(cursor, insert_query, values)
        connection.commit()
        print(f"Successfully inserted {len(processed_data)} records into the database.")
    except Exception as e:
        print(f"Error while inserting data: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()

def process_pdfs_with_db(pdf_files: List[str], conn_params: Dict, chunk_size: int = 1024, overlap: int = 20, use_threads: bool = True):
    """
    Processes multiple PDF files, generates embeddings, and stores results in a database.

    Args:
        pdf_files (List[str]): List of PDF file paths.
        conn_params (Dict): Database connection parameters.
        chunk_size (int): Size of each text chunk.
        overlap (int): Overlap between consecutive chunks.
        use_threads (bool): Use ThreadPoolExecutor if True, otherwise ProcessPoolExecutor.
    """
    executor_class = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
    with executor_class() as executor:
        for processed_data in executor.map(lambda pdf: list(process_single_pdf(pdf, chunk_size, overlap)), pdf_files):
            store_pdf_data_to_db(processed_data, conn_params)
            print(f"Stored data for {len(processed_data)} chunks in the database.")