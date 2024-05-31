import streamlit as st
from key import cohere_api_key
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from key import cohere_api_key

# Cohere Model
from langchain_community.chat_models import ChatCohere

class CustomChatCohere(ChatCohere):
    def _get_generation_info(self, response):
        # Custom handling of generation info
        generation_info = {}
        if hasattr(response, 'token_count'):
            generation_info["token_count"] = response.token_count
        # Add other attributes if needed
        return generation_info

    def get_num_tokens(self, text: str) -> int:
        # Specify the model explicitly
        response = self.client.tokenize(text=text, model=self.model)
        return len(response.tokens)

llm = CustomChatCohere(cohere_api_key=cohere_api_key)

st.title('News Research Tool')
st.sidebar.title('News Article URLs')

urls = []
progress = st.empty()
for i in range(3):
    url = st.sidebar.text_input(f'Enter Url {i+1}')
    urls.append(url)

process_url = st.sidebar.button('Process URLs')
if process_url:
    # load data
    progress.text('URLs Loading ...Started...🟩🟩🟩🟩🟩🟩')
    from langchain_community.document_loaders import UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    url_data = loader.load()

    # split data
    progress.text('Text Splitting ...Started...🟩🟩🟩🟩🟩🟩')
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',' ','.'],
        chunk_size=1000
    )
    splits = splitter.split_documents(url_data)

    # vector embed and stored
    progress.text('Embedding Vector ...Started...🟩🟩🟩🟩🟩🟩')
    embedding = CohereEmbeddings(cohere_api_key=cohere_api_key)
    vector_index = FAISS.from_documents(splits, embedding)
    vector_index.save_local('faiss_store')
    progress.text('Vector Index Saved ✅')
    import time
    time.sleep(3)
    progress.text('Done✅')

query = st.text_input('Question: ')
if query:
    vector_index = FAISS.load_local("faiss_store", CohereEmbeddings(cohere_api_key=cohere_api_key), allow_dangerous_deserialization=True)
    # Retrieve information from the saved vector embeddings
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
    result = chain({'question': query}, return_only_outputs=True)
    # {"answer":"","sources":[]}
    st.header('Answer')
    st.write(result['answer'])

    # Display Sources
    sources = result.get('sources','')
    if sources:
        st.subheader('Sources:')
        sources_list = sources.split('\n')
        for source in sources_list:
            st.write(source)
    
