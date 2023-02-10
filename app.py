import streamlit as st
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pinecone

pinecone.init(api_key=st.secrets["pinecone_api"], environment=st.secrets["pinecone_env"])
index = pinecone.Index("uuvector")

def get_data(data):
    about = []
    name  = []
    page  = []
    for result in data['matches']:
        about.append(result.metadata['about'])#['metadata']
        name.append(result.metadata['name'])
        page.append(int(result.metadata['page']))
    return about,name,page

# embeddings
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
openai.api_key = st.secrets["api"]
 
#------------------------------------------------------------------------------------------
#LAYOUT
st.title("Selamat Datang di PENETIKUMSIA")
st.header("Penelusuran Semantik Hukum Indonesia versi Undang-Undang")
st.header("Hadirin Sekalian!! ⚖️ :memo:")

#Input
st.subheader('Apa yang ingin anda cari?')
prompt = st.text_input('Masukan prompt di sini')
#Doing cosine

if prompt != "":
    embedding = get_embedding(prompt, engine='text-embedding-ada-002')
    results = index.query(vector=embedding,
                          top_k=20,
                          include_values=False,
                          include_metadata=True
                         )

    about,name,page = get_data(results)
    
    Data = {'UU'      : about,   #change p4ompt to 20
            'Tentang' : name,     
            'Halaman' : page}
        
    st.table(Data)