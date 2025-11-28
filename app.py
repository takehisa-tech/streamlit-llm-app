from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ---------------------------------------
# LLMに問い合わせる関数（必須要件）
# ---------------------------------------
def ask_llm(user_text: str, expert_type: str) -> str:
    """入力テキストと専門家タイプを受け取り、LLMの回答を返す関数"""

    # 専門家の種類によってシステムメッセージを切り替える
    if expert_type == "A（AIエンジニア）":
        system_message = "あなたは高度な問題解決能力を持つAIエンジニアです。専門的かつ実装に役立つ助言をしてください。"
    elif expert_type == "B（ビジネスコンサルタント）":
        system_message = "あなたはロジカルで経営戦略に精通したビジネスコンサルタントです。理由を伴った提案を行ってください。"
    else:
        system_message = "あなたは丁寧でフレンドリーなアシスタントです。"

    # プロンプトテンプレート（Lesson8 形式）
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input_text}")
    ])

    # OpenAI Chat Model
    model = ChatOpenAI(model="gpt-4o-mini")  # 任意モデル。Streamlit Cloudでも動作。

    # チェーン生成
    chain = prompt | model

    # 実行
    response = chain.invoke({"input_text": user_text})

    return response.content


# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🧠 LangChain × Streamlit Demo App")

st.write("""
### 📘 アプリ概要
このアプリでは、入力フォームにテキストを入力し、  
選択した専門家の視点からLLMが回答します。  
LangChain を用いてプロンプトを構築し、回答内容を表示します。

#### 🔧 操作方法
1. **専門家の種類** をラジオボタンから選ぶ  
2. **テキストを入力**  
3. **「送信」ボタンを押す**  
4. LLM からの回答が画面下に表示されます
""")

# ラジオボタン（専門家選択）
expert_choice = st.radio(
    "LLMの専門家タイプを選択してください：",
    ("A（AIエンジニア）", "B（ビジネスコンサルタント）")
)

# 入力フォーム
user_input = st.text_area("テキストを入力してください：", height=180)

# ボタン
if st.button("送信"):
    if user_input.strip():
        with st.spinner("LLMに問い合わせ中..."):
            result = ask_llm(user_input, expert_choice)
        st.success("回答:")
        st.write(result)
    else:
        st.warning("入力テキストを入力してください。")
