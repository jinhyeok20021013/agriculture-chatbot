import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- 1. 환경 설정 및 API 키 확인 ---
def setup_environment():
    """Streamlit Secrets에서 API 키를 설정하고 로드합니다."""
    # OpenAI API 키는 Streamlit Secrets(또는 os.environ)에 'OPENAI_API_KEY'로 저장되어 있어야 합니다.
    # 사용자에게 API 키 입력을 유도하는 방법으로 대체합니다.
    openai_api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요:", type="password")
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True
    return False

# --- 2. RAG 시스템 핵심 함수 ---
def create_rag_chain(pdf_path):
    """
    PDF 파일 경로를 받아 LangChain을 이용한 RAG 체인을 생성합니다.
    """
    st.info("📚 자료(PDF)를 읽고 AI가 학습(색인)하는 중...", icon="⏳")
    
    # 1. 문서 로드 (PyPDFLoader 사용)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. 텍스트 분할 (청크 생성)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. 임베딩 및 벡터 저장소 생성 (FAISS 사용)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 4. LLM 모델 설정 (ChatOpenAI)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # 5. RAG 체인 생성 (RetrievalQA)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False # 소스 문서 반환 여부
    )
    
    st.success("✅ 학습 완료! 이제 질문할 수 있습니다.")
    return qa_chain

# --- 3. Streamlit 앱 메인 로직 ---
def main():
    st.set_page_config(page_title="농업회사법인 RAG 챗봇", layout="wide")
    st.title("🌱 농업회사법인 및 농지법 전문 AI 챗봇")
    st.markdown("---")

    # 세션 상태 초기화 (RAG 체인 및 API 키 상태 저장)
    if 'rag_chain' not in st.session_state:
        st.session_state['rag_chain'] = None
    if 'api_key_valid' not in st.session_state:
        st.session_state['api_key_valid'] = False

    # API 키 설정
    st.session_state['api_key_valid'] = setup_environment()
    
    # 파일 업로드 (여기서는 Streamlit을 통해 파일을 받아 바로 처리)
    # 실제 배포 시에는 GitHub에 업로드된 파일을 직접 로드하는 코드로 변경해야 합니다.
    uploaded_file = st.sidebar.file_uploader(
        "📝 농업 관련 PDF 자료를 업로드하세요.",
        type=["pdf"],
        disabled=not st.session_state['api_key_valid']
    )

    if uploaded_file and st.session_state['api_key_valid']:
        # 업로드된 파일을 임시 저장하고 경로를 넘겨줌
        with open("uploaded_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # RAG 체인 생성
        if st.session_state['rag_chain'] is None:
            st.session_state['rag_chain'] = create_rag_chain("uploaded_doc.pdf")
            
        # --- 채팅 인터페이스 ---
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "농업회사법인 설립, 농지 취득 등 궁금한 점을 질문해 주세요!"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("AI가 자료를 검색하고 답변을 생성하는 중..."):
                    # RAG 체인 호출
                    response = st.session_state['rag_chain'].invoke(prompt)
                    st.markdown(response['result'])
                
                st.session_state.messages.append({"role": "assistant", "content": response['result']})

    elif not st.session_state['api_key_valid']:
        st.warning("🔑 계속하려면 OpenAI API Key를 사이드바에 입력해 주세요.")
    else:
        st.warning("업로드된 PDF 파일이 없습니다. 농업 법규 자료(PDF)를 업로드하세요.")

if __name__ == "__main__":
    main()
