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


        ################################### Hugging Face parameters#######################################
        #         param cache_folder: str | None = None
        # Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.

        # param encode_kwargs: Dict[str, Any] [Optional]
        # Keyword arguments to pass when calling the encode method for the documents of the Sentence Transformer model, such as prompt_name, prompt, batch_size, precision, normalize_embeddings, and more. See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode

        # param model_kwargs: Dict[str, Any] [Optional]
        # Keyword arguments to pass to the Sentence Transformer model, such as device, prompts, default_prompt_name, revision, trust_remote_code, or token. See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer

        # param model_name: str = 'sentence-transformers/all-mpnet-base-v2'
        # Model name to use.

        # param multi_process: bool = False
        # Run encode() on multiple GPUs.

        # param query_encode_kwargs: Dict[str, Any] [Optional]
        # Keyword arguments to pass when calling the encode method for the query of the Sentence Transformer model, such as prompt_name, prompt, batch_size, precision, normalize_embeddings, and more. See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode

        # param show_progress: bool = False
        # Whether to show a progress bar.

        # async aembed_documents(texts: list[str]) → list[list[float]]
        # Asynchronous Embed search docs.

        # Parameters
        # :
        # texts (list[str]) – List of text to embed.

        # Returns
        # :
        # List of embeddings.

        # Return type
        # :
        # list[list[float]]

        # async aembed_query(text: str) → list[float]
        # Asynchronous Embed query text.

        # Parameters
        # :
        # text (str) – Text to embed.

        # Returns
        # :
        # Embedding.

        # Return type
        # :
        # list[float]

        # embed_documents(texts: List[str]) → List[List[float]][source]
        # Compute doc embeddings using a HuggingFace transformer model.

        # Parameters
        # :
        # texts (List[str]) – The list of texts to embed.

        # Returns
        # :
        # List of embeddings, one for each text.

        # Return type
        # :
        # List[List[float]]

        # embed_query(text: str) → List[float][source]
        # Compute query embeddings using a HuggingFace transformer model.

        # Parameters
        # :
        # text (str) – The text to embed.

        # Returns
        # :
        # Embeddings for the text.

        # Return type
        # :
        # List[float]
        ##################################################################################################################


        self.provider=provider
        self.framework=framework
        self.device={'device': device}
        self.normalize_embeddings={'normalize_embeddings':normalize_embeddings}
        

        if self.provider=='HF':
            if self.framework=='langchain':

                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs=self.device,
                    encode_kwargs=self.normalize_embeddings,
                    )
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
        




            


        



    


        