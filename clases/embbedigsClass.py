from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



class Embbedings:

    def __init__(self, provider:str, framework:str, api_key=None , device:str='cpu', normalize_embeddings:bool=False) ->None:


        # provider= HF | OAI
        # HF=> HuggingFace free model
        # OAI => OpenAI pay model
        #
        # framework= langchain | llamaindex
        #
        # api_key= openAi api key ->only OAI


        self.provider=provider
        self.framework=framework
        self.device={'device': device}
        self.normalize_embeddings={'normalize_embeddings':normalize_embeddings}
        

        if provider=='HF':
            if framework=='langchain':

                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            else:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5"
                )
        elif provider=='OAI':
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
            # global
            Settings.embed_model = OpenAIEmbedding()

    def apply_embeding_document(self,text):
        if self.framework=='langchain':
            
            return self.embeddings.embed_documents([text])
        




            


        



    


        