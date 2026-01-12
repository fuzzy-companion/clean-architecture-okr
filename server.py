from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

load_dotenv()
app = Flask(__name__)

# ---------------------------
# FEATURE SCHEMA, PARSER & PROMPT
# ---------------------------

class FileOutput(BaseModel):
    path: str = Field(description="File path")
    content: str = Field(description="File content")


class FeatureOutput(BaseModel):
    files: List[FileOutput]


feature_parser = JsonOutputParser(pydantic_object=FeatureOutput)

feature_prompt = PromptTemplate(
    template="""
You are an expert Flutter Clean Architecture code generator.

Task:
Generate a complete Flutter feature following Clean Architecture principles based on the user’s input specification.

Feature description:
{spec}

Guidelines:
- Follow Clean Architecture layers strictly: **data**, **domain**, and **presentation**
- Use **BLoC pattern** for state management
- Ensure **separation of concerns** using **use cases** and **repositories**
- Always name the root folder using the **feature name**
  (e.g., "login" → "lib/features/login/")
- Do **not** include:
  - main.dart
  - pubspec.yaml
  - lib/ or features/ or feature/ root folders
  - Dependency injection or service locator setup
- Do **not** use external packages:
  - equatable
  - either
- Naming conventions:
  - Classes: UpperCamelCase with first two characters as feature acronym
  - Files: snake_case.dart
  - imports should use recommend package path structure
- Return **only** the generated files and their content
- Ensure output is in **valid JSON** format

Output format:
{format_instructions}
""",
    input_variables=["spec"],
    partial_variables={
        "format_instructions": feature_parser.get_format_instructions()
    },
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )

llm = ChatOllama(model="llama3.2:latest", temperature=0)

feature_chain = feature_prompt | llm | feature_parser

@app.route("/generate", methods=["POST"])
def generate_feature():
    try:
        data = request.get_json()
        user_prompt = data.get("input")

        if not user_prompt:
            return jsonify({"error": "Missing 'input' field"}), 400

        feature_result = feature_chain.invoke({"spec": user_prompt})

        return jsonify(feature_result), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "GenAI Flutter API is running ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)