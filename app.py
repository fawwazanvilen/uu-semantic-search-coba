import streamlit as st
import pickle
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

# dir = 'data_merged_all_01.pickle' # change this to relevant source
dir = 'data_merged_test_small.pickle' # test

with open(dir, 'rb') as file:
    # Deserialize the DataFrame using pickle.load()
    data_merged_all = pickle.load(file) # should be in pandas

# embeddings
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
openai.api_key = st.secrets["api"]
 
def search_reviews(df, description, n=20, pprint=True):
   embedding = get_embedding(description, engine='text-embedding-ada-002')
   df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

#------------------------------------------------------------------------------------------
#LAYOUT
st.title("Selamat Datang di PENESEMUNDIA")
st.header("Penelusuran Semantik Undang-Undang Indonesia")
st.header("Hadirin Sekalian!! ⚖️ :memo:")

#Input
st.subheader('Apa yang ingin anda cari?')
prompt = st.text_input('Masukan prompt di sini')

#Doing cosine

if st.button("Cari!"):
    search = search_reviews(data_merged_all, prompt, n=20)
    
    Data = {'UU'      : search['pdf_names'].values[:20],   #change prompt to 20
            'Tentang' : search['tentang'].values[:20],     
            'Halaman' : search['page_number'].values[:20]+1}
        
    st.table(Data)