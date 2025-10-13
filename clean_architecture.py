from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json, os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define output schema
response_schemas = [
    ResponseSchema(
        name="files",
        description="List of files with path and Dart content. Each element is an object with { path, content }"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Step 2: Define prompt
prompt = PromptTemplate(
    template="""
You are a Flutter project generator.
Create a clean-architecture feature based on the following spec:

{feature_spec}

Output only valid JSON matching this format:
{format_instructions}
""",
    input_variables=["feature_spec"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chain = prompt | llm

# Example input spec
feature_spec = {
    "feature_name": "auth",
    "entity": {
        "name": "User",
        "fields": [
            {"name": "id", "type": "String"},
            {"name": "name", "type": "String"},
            {"name": "email", "type": "String"}
        ]
    },
    "use_cases": ["get_current_user"],
    "datasource_endpoints": [
        {"name": "fetchCurrentUser", "path": "/me", "method": "GET"}
    ],
    "state_management": "bloc"
}

# Step 4: Run with invoke()
response = chain.invoke({"feature_spec": json.dumps(feature_spec)})

# Some LLMs return `AIMessage` → extract `.content`
raw_output = response.content if hasattr(response, "content") else response

parsed = parser.parse(raw_output)

# Step 5: Write files
for file in parsed['files']:
    path = file["path"]
    content = file["content"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Feature generated!")
