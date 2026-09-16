"""
StrOutputParser, JsonOutputParser 
are some of the output parsers that can be used to parse the output of a language model. 
The output parser is responsible for taking the raw output of the language model and converting it into a structured format that can be used by the application. 
The output parser can also be used to validate the output of the language model and ensure that it meets certain criteria.

Eg:
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions in 20 words."),
    ("human", "{question}")
])
llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)
parser = StrOutputParser() 

chain = prompt | llm | parser

response = chain.invoke({"question": "What is LangGraph?"})

# response will be a string containing the answer to the question.

the recommended output parser is pydantic output parser, which uses pydantic models to validate and parse the output of the language model.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()

class Movie(BaseModel):
    title: str = Field( description="The title of the movie")
    year: int = Field( description="The year the movie was released")
    genre: str = Field( description="The genre of the movie")
    
llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)
llm_with_structured_output = llm.with_structured_output(Movie)
response = llm_with_structured_output.invoke("What is the movie 'Inception' about?")
print(response)  # title='Inception' year=2010 genre='Science Fiction'
