from langchain.text_splitter import RecursiveCharacterTextSplitter, Language, MarkdownTextSplitter, PythonCodeTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter,MarkdownNodeParser,CodeSplitter,SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class Chunker:
    def __init__ (self, framework:str, provider:str='HF')->None:
        # provider= HF | OAI
        # HF=> HuggingFace free model
        # OAI => OpenAI pay model
        # framework= langchain | llamaindex

        self.framework=framework
        self.provider=provider

    def character_chunker(self, chunk_size:int,chunk_overlap:int, text:str ,separators=['\n\n', '\n', ' ', ''] ):

        if self.framework=='langchain':

            text_aplitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
            nodes=text_aplitter.create_documents([text])
        else :
            node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            nodes = node_parser.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )

        return nodes
    
    def document_specific_chunker(self, chunk_size:int, chunk_overlap:int, text:str):

        if self.framework=='langchain':

            splitter=MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            return splitter.create_documents([text])
        else:
            parser = MarkdownNodeParser()

            return parser.get_nodes_from_documents(text)

    def python_chunker(self, text):

        if self.framework=='langchain':
            splitter=PythonCodeTextSplitter()

            return splitter.create_documents([text])
        else:
            splitter = CodeSplitter(
            language="python",
            chunk_lines=40,  # lines per chunk
            chunk_lines_overlap=15,  # lines overlap between chunks
            max_chars=1500,  # max chars per chunk
        )
        nodes = splitter.get_nodes_from_documents(text)
    

    def language_spliter(self, language, text):

#           CPP = 'cpp'
            # GO = 'go'
            # JAVA = 'java'
            # KOTLIN = 'kotlin'
            # JS = 'js'
            # TS = 'ts'
            # PHP = 'php'
            # PROTO = 'proto'
            # PYTHON = 'python'
            # RST = 'rst'
            # RUBY = 'ruby'
            # RUST = 'rust'
            # SCALA = 'scala'
            # SWIFT = 'swift'
            # MARKDOWN = 'markdown'
            # LATEX = 'latex'
            # HTML = 'html'
            # SOL = 'sol'
            # CSHARP = 'csharp'
            # COBOL = 'cobol'
            # C = 'c'
            # LUA = 'lua'
            # PERL = 'perl'
            # HASKELL = 'haskell'
            # ELIXIR = 'elixir'
            # POWERSHELL = 'powershell'}
        if self.framework=='langchain':
            splitter=RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=450, chunk_overlap=4)

            return splitter.create_documents([text]) 
        else:
            splitter = CodeSplitter(
                language=language,
                chunk_lines=40,  # lines per chunk
                chunk_lines_overlap=15,  # lines overlap between chunks
                max_chars=1500,  # max chars per chunk
            )
            return splitter.get_nodes_from_documents(text)
        
    
    def semantic_chunker(self,text, breakpoint_threshold_type:str):       
        if self.framework=='langchain':
            #breakpoint_threshold_type:
            #  percentile=The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.
            #  standard_deviation= In this method, any difference greater than X standard deviations is split
            #  interquartile= In this method, the interquartile distance is used to split chunks.
            if self.provider=='HF':
                text_splitter = SemanticChunker(HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"))
            else:
                text_splitter = SemanticChunker(OpenAIEmbedding())
            return  text_splitter.create_documents([text] , breakpoint_threshold_type=breakpoint_threshold_type)
        else :
            if self.provider=='HF':
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            else:   
                embed_model = OpenAIEmbedding()
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
                )
            return splitter.get_nodes_from_documents(text)