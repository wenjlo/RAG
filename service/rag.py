import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from repository.chroma.client import Client
from asset.Prompt import Prompt

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cert/cdp-rd-vertex-ai.json'


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAG(Client):
    def __init__(self):
        super().__init__()
        self.llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.prompt = Prompt()

    def predict(self, collection_name, question):
        db_connection = Chroma(client=self.client, collection_name=collection_name,
                               embedding_function=self.embedding_model)
        retriever = db_connection.as_retriever(search_kwargs={"k": 5})
        # Debug for retriever documents:
        # print(retriever.invoke(
        #     input=question,
        #
        # ))
        chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt.prompt_template(collection_name) | self.llm_model | StrOutputParser()
        )

        response = chain.invoke(question)
        return response
