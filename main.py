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
    "description": """Call this to generate Python code that uses pandas for data analysis. The code should assist in understanding metrics or performing tasks such as summarizing data, finding correlations, handling missing values, etc. Example user inquiries could be 'show summary statistics for all columns', 'find correlation between age and salary', or 'find a mean of prices in the dataset'. Should return the desired metrics as a string.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The code to execute on a pandas df named 'df'. Should be written in Python and executable. Must end in a print() statement."
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
    },
    {
    "name": "finalize_response",
    "description": """Call this after generating all of the needed information for an appropriate repsonse. If this tool is to be called, it should be the last tool called as it ends the chain-of-thought and synthesizes what has been generated so far.

    It will ultimately return a Vega-lite spec that has an human-like explanation in the 'description' field.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "vega_spec": {
                "type": "dict",
                "description": "The final Vega-lite spec for data visualization. Must have a 'description' field."
            },
            "desc": {
                "type": "string",
                "description": "The final response to give to the user. May include information about the metrics."
            }
        },
        "required": ["vega_spec", "desc"],
        "additionalProperties": False
    }
    },
]


def create_vega_spec(query):
    response = client.chat.completions.create( # Chat
        messages=[
            
        ],
        model="gpt-4o",
        tools=tools
    )
    return f""

def create_analysis_code(query):
    return f""

def synthesize_final_ans(vega_spec, desc):
    vega_spec['description'] = desc
    
    return f""


tool_map = {
    'generate_chart': create_vega_spec,
    'analyze_data': create_analysis_code,
    'finalize_response': synthesize_final_ans
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

# Helper functions
#print msg in red, accept multiple strings like print statement
def print_red(*strings):
  print('\033[91m' + ' '.join(strings) + '\033[0m')

# print msg in blue, , accept multiple strings like print statement
def print_blue(*strings):
  print('\033[94m' + ' '.join(strings) + '\033[0m')

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query
    
def execute_panda_dataframe_code(code):
    """
    Execute the given python code and return the output. 
    References:
    1. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/utilities/python.py
    2. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/tools/python/tool.py
    """
     # Save the current standard output to restore later
    old_stdout = sys.stdout
    # Redirect standard output to a StringIO object to capture any output generated by the code execution
    sys.stdout = mystdout = StringIO()
    try:
		    # Execute the provided code within the current environment
        cleaned_command = sanitize_input(code)
        exec(code)
        
        # Restore the original standard output after code execution
        sys.stdout = old_stdout
				
				# Return any captured output from the executed code
        return mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return repr(e)

# Endpoint to interact with OpenAI API via LangChain
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    try:
        df = pd.DataFrame((request.sample))
        messages=[
            {
                "role": "system",
                "content": react_function_calling_prompt
            },
            {
                "role": "user",
                "content": f"Here are the headers: {', '.join(request.headers)}"
            },
            {
                "role": "user",
                "content": f"""Here is a subsection of the csv to better contextualize your answers: {df.head(20)} 
                """
            },
            {
                "role": "user",
                "content": f"""

                Your final Vega-Lite specification should at least have:
                - A "description" field that summarizes what the specification is.
                - The x-axis should be labeled.
                - The y-axis should be labeled.
                - Be in JSON format.
                - No URL field.

                Give me a Vega-Lite (JSON) specification that fulfills the following request: {request.prompt}"""
            },
        ]

        i = 0
        while i < 8: # Max of 8 iterations, change if needed
            response = client.chat.completions.create( # Chat
                messages=messages,
                model="gpt-4o",
                tools=tools
            )

            if response.choices[0].message.content:
                print_red(response.choices[0].message.content) 
            
            messages.append(response.choices[0].message)
            for tool_call in response.choices[0].message.tool_calls:
                print_blue('calling:'+tool_call.function.name)            
                
                # call the function
                arguments = json.loads(tool_call.function.arguments)
                function_to_call = tool_map[tool_call.function.name]
                result = function_to_call(**arguments) # save outcome

                # create a message containing the tool call result
                result_content = json.dumps({
                    **arguments,
                    "result": result
                })
                function_call_result_message = {
                    "role": "tool",
                    "content": result_content,
                    "tool_call_id": tool_call.id
                }
                print_blue('action result:' + result_content)

                # save the action outcome for LLM
                messages.append(function_call_result_message)
                
                # Finishing response, loop ends 
                if tool_call.function.name == 'finalize_response':
                    i += 8
                    break

                


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