import streamlit as st
import pickle
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pinecone

# useless pickles
# # dir = 'data_merged_all_01.pickle' # change this to relevant source
# dir = 'data_merged_test_small.pickle' # test

# with open(dir, 'rb') as file:
#     # Deserialize the DataFrame using pickle.load()
#     data_merged_all = pickle.load(file) # should be in pandas

# embeddings
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
openai.api_key = st.secrets["openai_api"]

# pinecone stuffs
pinecone.init(api_key=st.secrets["pinecone_api"], environment=st.secrets["pinecone_env"]) # define pinecone inits
# index = pinecone.Index(st.secrets["pinecone_index"]) # st.secrets seems to have problems if repeatedly called
index = pinecone.Index("uuvector")

# semantic search function 
def semanticsearch(query):
    embedding = get_embedding(query, engine='text-embedding-ada-002')
    results = index.query(vector=embedding, top_k=20, include_values=False, include_metadata=True)

    id = []
    about = []
    name  = []
    page  = []
    score = []
    for result in results['matches']:
        id.append(result.id)
        about.append(result.metadata['about'])
        name.append(result.metadata['name'])
        page.append(int(result.metadata['page']))
        score.append(result.score)

    df = pd.DataFrame(np.column_stack([id, about, name, page, score]),
                      columns=['id', 'about', 'name', 'page', 'score'])
    return df

#------------------------------------------------------------------------------------------
#LAYOUT
st.title("Selamat Datang di PENESEMUNDIA")
st.header("Penelusuran Semantik Undang-Undang Indonesia")
st.header("Hadirin Sekalian!! ⚖️ :memo:")

# if "cari" not in st.session_state:
#     st.session_state.cari = False

# def prompt_callback():
#     st.session_state.cari = True

#Input
st.subheader('Apa yang ingin anda cari?')
# prompt = st.text_input('Masukan prompt di sini', on_change=prompt_callback)
prompt = st.text_input('Masukan prompt di sini')

# st.session_state.cari = st.button("Cari!")

#Doing cosine
if st.button("Cari!") or prompt!="":
# if st.session_state.cari:
    # search = search_reviews(data_merged_all, prompt, n=20)
    search = semanticsearch(prompt)
    
    Data = {'UU'      : search['name'].values[:20],   #change prompt to 20
            'Tentang' : search['about'].values[:20],     
            'Halaman' : search['page'].values[:20]}
        
    st.table(Data)