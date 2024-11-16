from fastapi import FastAPI
import os
from dotenv import load_dotenv
from typing import Optional, Union
from pydantic import Field, BaseModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages.ai import AIMessage
import time
import json


# Load the environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('OPENAI_API'))


# Define evaluation schemas
descriptions = {
    "status": """
        This field will be one of the following values:
        - Incomplete or erroneous
            Meaning: Code is incomplete or it is producing errors (it cannot be run succesfully).
        - Incorrect but better
            Meaning: Code is incorrect but better than previous code snippet in terms of syntax, logic and correctness. This means the user is moving in right direction towards the correct code.
        - Incorrect and worse
            Meaning: Code is incorrect and worse than previous code snippet in terms of syntax, logic -  This means the user is moving in wrong direction away from the correct code.
        - Incorrect and same as previous code snippet
            Meaning: Code is incorrect and same as previous code snippet and there is no improvement in the code.
        - Partially correct
            Meaning: Code is partially correct. It covers some of the requirements from the question but not all.
        - Correct but can be improved
            Meaning: Syntax is correct, code does not produce any error on running it and output is as expected BUT the code can be improved (improvements can be in terms of code optimization, better logic, better variable names, better comments, etc.)
        - Correct
            Meaning: Syntax is correct, code does not produce any error on running it and output is as expected. Code is following the guidelines provided in the question. There is no redundancy in the code. Code is optimized and well written.
        """,

    "score": """
        An approximate score out of 100 as the quality of the code snippet provided by user. This score should be based on the quality of code snippet and how well it is following the guidelines provided in the question.
    """,

    "result": """
        - 1
            Meaning: The given code is better than the previous code snippet in terms of syntax, logic and correctness. This means the user is moving in right direction towards the correct code.
        - -1
            Meaning: The given code is worse than the previous code snippet in terms of syntax, logic or correctness. This means the user is moving in wrong direction away from the correct code.
    """,

    "comment": """
        One line comment showing the reason for the given score.
    """
}

class Evaluation(BaseModel):
    """Evaluation of the code provided by the user."""
    status: str = Field(description=descriptions["status"])
    score: int = Field(description=descriptions["score"])
    result: int = Field(description=descriptions["result"])
    comment: str = Field(description=descriptions["comment"])

class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")

class FinalResponse(BaseModel):
    final_output: Union[Evaluation, ConversationalResponse]

# Configure LLM for structured output
model_with_structured_output = llm.with_structured_output(FinalResponse)

# Define state and workflow
class AgentState(MessagesState):
    messages: list[BaseMessage]
    final_output: Optional[FinalResponse] = None

def call_model(state: AgentState) -> AgentState:
    response = model_with_structured_output.invoke(state["messages"])
    print("RESPONSE", response)
    return {
        "messages": state["messages"] + [AIMessage(content=str(response.final_output))],
        "final_output": response
    }

# Setup workflow
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Initialize memory and compile workflow
memory = MemorySaver()
llmApp = workflow.compile(checkpointer=memory)

# Helper functions
def init():
    system_message = SystemMessage(content="""
You are a python code quality evaluator.
1. You will receive a question in the next prompt.
2. Then you will receive code snippets from users periodically. 
3. You must evaluate the code quality and provide output.
4. Do not provide solutions, suggestions, or hints.
""")
    
    return get_output([system_message])

def get_output(messages: list[BaseMessage]):
    config = {"configurable": {"thread_id": "10202"}}
    output = llmApp.invoke({"messages": messages}, config)
    final_output = json.loads(output["final_output"].json())["final_output"]
    if output["messages"]:
        output["messages"][-1].pretty_print()
        print("FINAL OUTPUT", final_output)
    return output

def message_parser(messages: list[BaseMessage]):
    parsed_messages = []
    print(messages)
    for message in messages:
        if message.type == "system":
            parsed_messages.append(SystemMessage(content=message.content))
        elif message.type == "human":
            parsed_messages.append(HumanMessage(content=message.content))
        elif message.type == "ai":
            parsed_messages.append(AIMessage(content=message.content))
    return parsed_messages

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A Code Ecaluator API server using LangChain's Runnable interfaces",
)

# main api
@app.post("/evaluate/")
async def evaluate_code(messages: list[BaseMessage], user_code_snippet: str) -> dict:
    messages.append(HumanMessage(content=user_code_snippet))
    return get_output(messages)

@app.post("/init/")
async def init_conversation():
    return init()

class FeedQuestion(BaseModel):
    question: str
    messages: list[BaseMessage]

@app.post("/feed-question/")
async def feed_question(body: FeedQuestion):
    question = body.question
    messages = message_parser(body.messages)

    print("QUESTION", question)
    print("MESSAGES", messages)

    question_message = SystemMessage(content=f"Question: {question}")
    return get_output(messages + [question_message])

@app.get("/")
async def root():
    return {"message": "Welcome to the Code Evaluator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)