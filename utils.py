"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct
import pandas as pd  # Add pandas for CSV processing
import logging  # Add logging for debug messages


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def filter_csv_rows_by_department(chat_message, csv_path):
    """
    CSVファイルを読み込み、指定部署でフィルタリングしたテーブルをMarkdown形式で返す
    """
    import re
    logger = logging.getLogger(ct.LOGGER_NAME)
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        if '部署' not in df.columns:
            logger.error(f"CSV filtering failed: '部署'列が存在しません")
            return None
        # 部署名の抽出: 正規表現 or 実データからマッチ
        match = re.search(r"部署が(.+?)の", chat_message)
        if match:
            dept_name = match.group(1)
        else:
            # ユニークな部署名からチャットメッセージに含まれるものを探す
            dept_name = None
            for dept in df['部署'].unique():
                if dept and dept in chat_message:
                    dept_name = dept
                    break
            if not dept_name:
                # フォールバック: '部署'以降の文字を削除
                dept_name = chat_message.replace("部署", "").split()[0]
        logger.info(f"Filtering CSV for department: {dept_name}")
        filtered = df[df['部署'] == dept_name]
        if filtered.empty:
            return f"部署「{dept_name}」の従業員情報が見つかりませんでした。"
        return filtered.to_markdown(index=False)
    except Exception as e:
        logger.error(f"CSV filtering failed: {e}")
        return None


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # CSV結合ドキュメントの場合、社内問い合わせモードでのみアプリ側で直接処理
    if st.session_state.mode == ct.ANSWER_MODE_2:
        retriever = st.session_state.retriever
        if retriever and hasattr(retriever, 'get_relevant_documents'):
            try:
                docs = retriever.get_relevant_documents(chat_message)
            except Exception:
                docs = []
            if docs and docs[0].metadata.get('source') == 'combined_csv':
                csv_path = docs[0].metadata.get('original_sources', '')
                result = filter_csv_rows_by_department(chat_message, csv_path)
                if result:
                    return {'answer': result, 'context': docs}

    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    if st.session_state.mode == ct.ANSWER_MODE_1:
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})

    # ページ番号付与処理を削除
    for document in llm_response["context"]:
        pass

    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response