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
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import constants as ct
import csv


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

############################################################
# 変数定義
############################################################
seach_kwargs_num = 5  # RAGのRetrieverで取得するドキュメント数
chunk_size_num = 1024  # チャンク分割のサイズ
chunk_overlap_num = 256  # チャンク分割のオーバーラップサイズ

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # CSVファイルのみを結合対象とし、他のドキュメントは除外
    csv_docs = [doc for doc in docs_all if doc.metadata.get("source", "").endswith(".csv")]
    if csv_docs:
        combined_doc = combine_csv_rows_into_single_document(csv_docs)
        docs_all = [combined_doc]

    # CSVファイルの場合、各行にメタデータを付与
    if csv_docs:
        docs_all = add_metadata_to_csv_rows(docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size_num,
        chunk_overlap=chunk_overlap_num,
        separator="\n"
    )

    # チャンク分割を実施（CSV結合ドキュメントは分割しない）
    if docs_all and docs_all[0].metadata.get("source") == "combined_csv":
        splitted_docs = docs_all
    else:
        splitted_docs = text_splitter.split_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # seach_kwargs_numを動的に調整
    chat_message = st.session_state.get("last_user_message", "")
    adjusted_seach_kwargs_num = adjust_seach_kwargs_num_based_on_csv(chat_message)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": adjusted_seach_kwargs_num})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    # デバッグ用: 読み込んだデータソースの数をログに記録
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"Total documents loaded: {len(docs_all)}")

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    # デバッグ用: Webページから読み込んだデータソースの数をログに記録
    logger.info(f"Total web documents loaded: {len(web_docs_all)}")

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()

        # PDFの場合、ページ番号をメタデータに追加
        if file_extension == ".pdf":
            import fitz  # PyMuPDF
            pdf_document = fitz.open(path)
            for i, doc in enumerate(docs):
                doc.metadata["page"] = f"ページNo.{i + 1}"
            pdf_document.close()

        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


def adjust_seach_kwargs_num_based_on_csv(chat_message):
    """
    CSV参照時に動的にseach_kwargs_numを調整する

    Args:
        chat_message: ユーザー入力値

    Returns:
        動的に調整されたseach_kwargs_numの値
    """
    # CSV参照が必要かどうかを判定（例: "CSV"というキーワードが含まれる場合）
    if "CSV" in chat_message:
        # CSVの行数に基づいてseach_kwargs_numを調整（仮に10行とする）
        return 10
    
    # デフォルト値を返す
    return seach_kwargs_num


def add_metadata_to_csv_rows(docs):
    """
    CSVの各行にメタデータ（行番号、列名）を付与する

    Args:
        docs: CSVLoaderで読み込んだドキュメントリスト

    Returns:
        メタデータが付与されたドキュメントリスト
    """
    updated_docs = []
    for i, doc in enumerate(docs):
        # 行番号をメタデータに追加
        doc.metadata["row_number"] = i + 1
        # 列名をメタデータに追加（仮に列名が取得可能な場合）
        if "columns" in doc.metadata:
            doc.metadata["columns"] = doc.metadata["columns"]
        updated_docs.append(doc)
    return updated_docs


def combine_csv_rows_into_single_document(docs):
    """
    CSVの全行を1つのドキュメントに結合する
    """
    # CSVを直接読み込んで全行を結合
    csv_path = docs[0].metadata.get("source") if docs else None
    content_lines = []
    if csv_path:
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    content_lines.append(", ".join(row))
        except Exception:
            # エラー時は既存のdocsから結合
            for doc in docs:
                if doc.page_content:
                    content_lines.append(doc.page_content)
    combined_content = "\n".join(content_lines)
    # デバッグ用ログ
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"Combined CSV read directly: {len(content_lines)} lines. Content length: {len(combined_content)} chars.")
    combined_metadata = {
        "source": "combined_csv",
        "row_count": len(content_lines) - 1,  # ヘッダーを除く行数
        "original_sources": csv_path or ""
    }
    return Document(page_content=combined_content, metadata=combined_metadata)