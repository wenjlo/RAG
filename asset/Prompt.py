from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import datetime


class Prompt:
    def __init__(self):
        self.comment_system_prompt = """
    You are a data analyst.
    Your job is to answer questions based on the questions and information provided.
    Provide brief data analysis comments.
    Need to list information.
    Need to show the information.
    If the question contains a date, the answer should also include a date.
    Payoff is from the member's perspective, negative numbers mean the company is making money .
        """
        self.ner_human_prompt = """ 
The original sentence is: {question}
 Use the new context below to rewrite the original sentence
{context}
Please do not provide any information that is irrelevant to the rewritten sentence.
Don't change the ordering of the sentence grammar, just insert the entity before the text.
Please answer the question in Chinese.
"""
        self.ner_prompt = f"""You are a smart and intelligent Named Entity Recognition (NER) system.
Your job is to recognize entities from sentence and insert entities in front of the words and
identify the date and add the year if the date does not include a year.
The entities that must be identified are <動物>
If an entity is found to already exist in the sentence, there is no need to insert entities in front of the words.
If the date in the sentence does not have a year, please enter {str(datetime.datetime.now().year)}. 
If it does have a year, please keep the original date without any modification.
Please do not provide any information that is irrelevant to the rewritten sentence and 
Do not change the order of the words in the sentence or delete words.
If you encounter words like today, yesterday, or last week, do not perform entity recognition on the date.
 """

        self.sql_human_prompt = """
The user's question is: {question}
Use the new context below to generate sql query
{context}
Construct a JSON response containing the following keys and values: 
'sql_type': mysql
'sql_query': the sql query.
The answer should be a valid JSON format.
Do not include newline characters. 
*  Please answer in pure JSON format without any prefix or suffix.
* "Please output the following information in JSON format, 
without any extra text or markup, such as ```json```."
"""
        self.sql_prompt = """You are a senior data engineer your job is to recognize the question and answer the question with sql query 
and generate JSON responses for specific data structures.
* "Please output the following information in JSON format, 
without any extra text or markup, such as ```json```."

Given a schema document and question from user,
you should answer the sql query based on the given schema document.

"""

    def prompt_template(self, job_name="ner"):
        if job_name == "ner":
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.ner_prompt),
                HumanMessagePromptTemplate.from_template(self.ner_human_prompt)
            ])
        elif job_name == "sql":
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.sql_prompt),
                HumanMessagePromptTemplate.from_template(self.sql_human_prompt)
            ])

        return chat_template