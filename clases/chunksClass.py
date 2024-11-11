from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunker, Language, MarkdownTextSplitter, PythonCodeTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

class Chunker:
    def __init__ (self):
        pass

    def character_chunker(chunk_size:int,chunk_overlap:int, text:str ,separators=['\n\n', '\n', ' ', ''] ):


        text_aplitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

        return text_aplitter.create_documents([text])
    
    def document_specific_chunker(chunk_size:int, chunk_overlap:int, text:str):

        splitter=MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        return splitter.create_documents([text])
    
    def python_chunker(text):

        splitter=PythonCodeTextSplitter()

        return splitter.create_documents([text])
    

    def language_spliter(language, text):

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

        splitter=RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=450, chunk_overlap=4)

        return splitter.create_documents([text]) 
    
    def semantic_chunker(text, breakpoint_threshold_type:str):       
       
        #breakpoint_threshold_type:
        #  percentile=The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.
        #  standard_deviation= In this method, any difference greater than X standard deviations is split
        #  interquartile= In this method, the interquartile distance is used to split chunks.
            
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        docs = text_splitter.create_documents([text] , breakpoint_threshold_type=breakpoint_threshold_type)
        print(docs[0].page_content)