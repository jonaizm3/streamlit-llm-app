import os  # osモジュールも後でAPIキーを取得するのに使うのでインポートしておきます
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得（取得できるかテスト表示もしてみましょう）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import httpx

# APIキーが設定されていない場合の早期エラーチェック
if not OPENAI_API_KEY:
    st.error("OpenAI APIキーが設定されていません。.envファイルを確認してください。")
    st.stop() # APIキーがない場合はアプリを停止
    

# LLMからの回答を取得する関数
def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    ユーザー入力と専門家の種類に基づいてLLMからの回答を取得する。

    Args:
        user_input (str): ユーザーが入力したテキスト。
        expert_type (str): ラジオボタンで選択された専門家の種類。

    Returns:
        str: LLMからの回答テキスト。
    """
    # 専門家の種類に応じてシステムメッセージを定義
    if expert_type == "心理カウンセラー":
        system_message_content = "あなたは経験豊富で共感力の高い心理カウンセラーです。利用者の悩みや感情に寄り添い、専門的な知識に基づいて具体的かつ建設的なアドバイスをしてください。"
    elif expert_type == "健康アドバイザー":
        system_message_content = "あなたは知識が豊富で信頼できる健康アドバイザーです。栄養、運動、生活習慣に関する質問に対し、科学的根拠に基づいた分かりやすい情報と実践的なアドバイスを提供してください。"
    else:
        # デフォルトまたは予期しない選択の場合
        system_message_content = "あなたは親切で有能なAIアシスタントです。"

    # ChatOpenAIモデルの初期化
    try:
        custom_httpx_client = httpx.Client(
            proxies=None,
            trust_env=False # 環境変数からのプロキシ設定読み込みを無効化
        )
        chat = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # 使用するモデルを指定
            temperature=0.7,            # 回答の多様性を調整
            openai_api_key=OPENAI_API_KEY, # APIキーを渡す
            http_client=custom_httpx_client # カスタムHTTPクライアントを指定
        )
    except Exception as e:
        st.error(f"OpenAIモデルの初期化中にエラーが発生しました: {e}")
        # ローカル実行時はコンソールにもエラーを出力するとデバッグしやすい
        print(f"Error initializing OpenAI model: {e}")
        return "モデルの初期化に失敗しました。設定を確認してください。"

    # LLMに渡すメッセージリストを作成
    messages = [
        SystemMessage(content=system_message_content), # システムメッセージ
        HumanMessage(content=user_input)              # ユーザーからの入力
    ]

    # LLMに問い合わせて回答を取得
    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"LLMからの回答取得中にエラーが発生しました: {e}")
        # ローカル実行時はコンソールにもエラーを出力
        print(f"Error getting response from LLM: {e}")
        return "回答の取得中にエラーが発生しました。しばらくしてからもう一度お試しください。"
  
# --- Streamlit UIの定義 ---

st.title("AI専門家チャット")

st.markdown("""
### アプリ概要
このアプリケーションでは、AIの専門家とチャットすることができます。
相談したい専門家のタイプを選び、下のフォームに質問や相談内容を入力してください。

### 操作方法
1.  **専門家を選択**: ラジオボタンから「心理カウンセラー」または「健康アドバイザー」を選びます。
2.  **相談内容を入力**: テキストエリアに、専門家に聞きたいことや相談したい内容を具体的に入力します。
3.  **送信**: 「回答を生成」ボタンをクリックすると、AI専門家からの回答が表示されます。
""")

# ラジオボタンで専門家の種類を選択
expert_options = ["心理カウンセラー", "健康アドバイザー"]
selected_expert = st.radio(
    "相談したい専門家のタイプを選んでください:",
    expert_options,
    index=0  # デフォルトの選択 (0なら最初のオプション「心理カウンセラー」)
)

# テキスト入力フォーム
user_query = st.text_area(f"{selected_expert}への相談内容を入力してください:", height=150)

# 送信ボタン
if st.button("回答を生成"):
    if not user_query.strip():
        st.warning("相談内容を入力してください。")
    else:
        with st.spinner(f"{selected_expert}が考えています..."):
            # LLMからの回答を取得
            llm_answer = get_llm_response(user_query, selected_expert)
            st.subheader(f"{selected_expert}からの回答:")
            st.write(llm_answer)
else:
    # ボタンが押される前（初期状態）に何か表示したい場合はここに書く
    # 例: st.info("上記に相談内容を入力し、「回答を生成」ボタンを押してください。")
    pass # 何も表示しない場合は pass でもOK
