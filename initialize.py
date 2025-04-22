"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LC_Document
import constants as ct
import csv

# 定数・設定値の一元管理
from constants import CHUNK_SIZE_NUM, CHUNK_OVERLAP_NUM, SEACH_KWARGS_NUM

############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

############################################################
# 関数定義
############################################################

def initialize():
    """
    初期化処理
    """
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    RAGのRetrieverを作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    if "retriever" in st.session_state:
        return

    docs_all = load_data_sources()
    csv_docs = [doc for doc in docs_all if doc.metadata.get("source", "").endswith(".csv")]
    if csv_docs:
        combined = combine_csv_rows_into_single_document(csv_docs)
        docs_all = [combined]

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        doc.metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}

    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE_NUM,
        chunk_overlap=CHUNK_OVERLAP_NUM,
        separator="\n"
    )

    if docs_all and docs_all[0].metadata.get("source") == "combined_csv":
        splitted = docs_all
    else:
        splitted = text_splitter.split_documents(docs_all)

    db = Chroma.from_documents(splitted, embedding=embeddings)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": SEACH_KWARGS_NUM})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def load_data_sources():
    """
    データソースの読み込み
    """
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"Total documents loaded: {len(docs_all)}")

    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        web_docs_all.extend(web_docs)
    docs_all.extend(web_docs_all)

    logger.info(f"Total web documents loaded: {len(web_docs_all)}")
    return docs_all


def recursive_file_check(path, docs_all):
    """
    データソースの読み込み
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み
    """
    file_extension = os.path.splitext(path)[1]
    file_name = os.path.basename(path)

    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()

        if file_extension == ".pdf":
            import fitz  # PyMuPDF
            pdf_document = fitz.open(path)
            for i, doc in enumerate(docs):
                doc.metadata["page"] = f"ページNo.{i + 1}"
            pdf_document.close()

        docs_all.extend(docs)


def adjust_string(s):
    """
    文字列調整
    """
    if type(s) is not str:
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    return s


def combine_csv_rows_into_single_document(docs):
    """
    CSVの全行を1つのドキュメントに結合
    """
    csv_path = docs[0].metadata.get("source") if docs else None
    content_lines = []
    if csv_path:
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    content_lines.append(", ".join(row))
        except Exception:
            for doc in docs:
                if doc.page_content:
                    content_lines.append(doc.page_content)
    combined_content = "\n".join(content_lines)
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"Combined CSV read directly: {len(content_lines)} lines. Content length: {len(combined_content)} chars.")
    combined_metadata = {
        "source": "combined_csv",
        "row_count": len(content_lines) - 1,
        "original_sources": csv_path or ""
    }
    return LC_Document(page_content=combined_content, metadata=combined_metadata)