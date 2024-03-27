import warnings

# Your code here

# To ignore all warnings, you can use the following line:
warnings.filterwarnings("ignore")

# Or, to ignore a specific category of warnings (e.g., DeprecationWarning):
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Rest of your code

# Remember to reset the warning filters if needed, especially if this script is part of a larger project.
# Resetting the filters can be done by calling warnings.resetwarnings().
import os
import boto3
import pickle
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
#from langchain.llms import HuggingFacePipeline, LlamaCpp

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

#-----------------------------------

raw_documents = TextLoader("./chapters.txt",autodetect_encoding=True).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, HuggingFaceEmbeddings())
print(db)

#-------------------------------------

# ----------------------------------------


query = "What are index funds? Describe in one line."
docs = db.similarity_search(query)
#print(docs[0].page_content)


#-----------------------------------------------

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = Bedrock(
        credentials_profile_name="default",
        model_id="meta.llama2-13b-chat-v1",
        model_kwargs={
            "temperature": 0.1,
            "top_p": 0.1,
            "max_gen_len": 128
        }
    )

"""
llm = LlamaCpp(
    model_path="./llama2/7B-chat-gguf/llama-2-7b-chat.ggufv3.q4_0.bin",
    temperature=0.1,
    max_tokens=128,
    top_p=0.1,
    n_ctx=1024,
    callback_manager=callback_manager,
    verbose=True, # verbose is required to pass to the callback manager
)

"""
# ----------------------------------------------------

print(llm(query))

# -------------------------

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True,
)

#-----------------------------------


output = rag_pipeline(query)


#---------------------------------------

for i, doc in enumerate(output["source_documents"]):
    print(f"Source Document {i + 1}:")
    print(doc.page_content)
    print()

#-----------------------------------------------



