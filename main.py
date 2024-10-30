from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import sys
from io import StringIO
import pandas as pd


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="client/build/", html=True), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://csci3360-assignment2.onrender.com/",
                   "*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str
    headers: list
    sample: list # can be entire csv, remember to process
    
class QueryResponse(BaseModel):
    response: dict   

### ReAct setup
# Tools + mapping
tools = [
    {
    "name": "generate_chart",
    "description": """Call this to generate a Vega-lite specification based on the user's inquiry for visualization. Requests can be short and should relate to the data. Example user inquiries could be 'show mpg over time' or 'cars by origins' if the csv is about cars. Should return a JSON object.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's query. Can be re-written to be more comprehensible for Vega-lite generation, which will be handled by OpenAI.",
            },
        },
        "required": ["query"],  
        "additionalProperties": False,
    }
    },{
    "name": "analyze_data",
    "description": """Call this to generate Python code that uses pandas for data analysis. The code should assist in understanding metrics or performing tasks such as summarizing data, finding correlations, handling missing values, etc. Example user inquiries could be 'show summary statistics for all columns', 'find correlation between age and salary', or 'identify missing data in the dataset'. Should return Python code as a string.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's plaintext query that will be translated into executale code. Can be re-written to be more specific for code generation, which will be handled by OpenAI."
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
    },
    {
    "name": "finalize_response",
    "description": """Call this after generating all of the needed information for an appropriate repsonse. If this tool is to be called, it should be the last tool called as it ends the chain-of-thought and synthesizes what has been generated so far.

    It will ultimately return a Vega-lite spec that has an appropriate 'description' field.

    """,
    "parameters": {
        "type": "object",
        "properties": {
            "vega_spec": {
                "type": "dict",
                "description": "The final Vega-lite spec for data visualization"
            },
        },
        "required": ["vega_spec"],
        "additionalProperties": False
    }
    },
    {
    "name": "reject_query",
    "description": """Call this function if the user's query is irrelevant to the data. Should return an empty Vega-lite spec whose 'description' field contains a reasoning for why the user is wrong and a suggestion of topics/queries that are acceptable based on the data. If this function is to be called, then it will be the last function called.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's query, which is needed for context."
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
    }
]

def create_vega_spec():
    return ""

def create_analysis_code():
    return ""


tool_map = {
    'generate_chart': get_data_overview,
    'analyze_data': execute_sql
}

react_function_calling_prompt = '''You are a helpful assistant.


You run in a loop of Thought, Action, Observation in the following manner to answer the user question.

Question: the input question you must answer

Thought: you should always think about what to do
Action: the action to take, should be one of tools provided
Action Input: the input to the action

You will be then call again with the result of the action.

Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

'''




# Endpoint to interact with OpenAI API via LangChain
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    try:
        # Call the OpenAI API via LangChain
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Here are the headers: {', '.join(request.headers)}"
                },
                {
                    "role": "user",
                    "content": f"""Here is the full csv to better contextualize your answers: {json.dumps(request.sample)} 
                    """
                },
                {
                    "role": "user",
                    "content": f"""

                    The Vega-Lite specification should at least have:
                    - A "description" field that summarizes what the specification is.
                    - The x-axis should be labeled.
                    - The y-axis should be labeled.
                    - Be in JSON format.
                    - No URL field.

                    Give me a Vega-Lite (JSON) specification that fulfills the following request: {request.prompt}"""
                },
            ],
            model="gpt-4o",
            temperature=0,
            response_format={ "type": "json_object" }
        )

        try:
            response_json = json.loads(chat_completion.choices[0].message.content.strip())
            return QueryResponse(response=response_json)
        except json.JSONDecodeError: # Response had faulty JSON code
            raise HTTPException(status_code=500, detail="Failed to decode JSON from OpenAI response.")
    except Exception as e: # Chat completion exception (failed to connect to OpenAI / timed out)
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def read_root():
    return FileResponse('client/build/index.html')