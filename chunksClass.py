from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__ (self):
        pass

    def caracter_chunker(chunk_size:int,chunk_overlap:int, text:str ,separators=['\n\n', '\n', ' ', ''] ):


        text_aplitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

        return text_aplitter.create_documents([text])