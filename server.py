from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# intent detection schema using Pydantic
class IntentSchema(BaseModel):
    intent: str = Field(description="Either 'create_feature' or 'modify_code'")

intent_parser = JsonOutputParser(pydantic_object=IntentSchema)
intent_prompt = PromptTemplate(
    template="""
You are an assistant that analyzes software development requests.

Classify the user's intent based on the input description below.

User input:
{prompt}

Possible intents:
- create_feature → user wants to generate a new feature or architecture
- modify_code → user wants to modify or update existing files

Output valid JSON only:
{format_instructions}

Output STRICTLY valid JSON (escaped properly). Do not include code fences, markdown, or text outside JSON.
""",
    input_variables=["prompt"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()},
)

# feature generation schema using Pydantic
class FileSchema(BaseModel):
    path: str = Field(description="File path")
    content: str = Field(description="File content")

class FeatureSchema(BaseModel):
    files: list[FileSchema] = Field(description="List of files to create. Each item: { path, content }")

feature_parser = JsonOutputParser(pydantic_object=FeatureSchema)

feature_prompt = PromptTemplate(
    template="""
You are an expert Flutter Clean Architecture code generator.

Task:
Generate a complete Flutter feature following Clean Architecture principles based on the user's input specification.

Feature description:
{spec}

Guidelines:
- Follow Clean Architecture layers strictly: **data**, **domain**, and **presentation**.
- Use **BLoC pattern** for state management (no other pattern until further notice).
- Ensure **separation of concerns** using **use cases** and **repositories**.
- Always name the root folder using the **feature name** (e.g., "login" → "lib/features/login/").
- Do **not** include:
  - `main.dart`
  - `pubspec.yaml`
  - Skip generating the lib directory and features directory. Place all new files directly under the existing features folder.
  - Dependency injection or service locator setup
- Do **not** use external packages such as:
  - `equatable`
  - `either`
- Follow naming conventions:
  - **Classes:** Use UpperCamelCase with the first two characters as the feature acronym (e.g., `UserProfile` → `UPUserProfileScreen`)
  - **Files:** Use snake_case (e.g., `user_profile_screen.dart`)
- Return **only** the generated files and their content — no explanations or comments.
- Ensure output is in **valid JSON** format.

Output format:
{format_instructions}

""",
    input_variables=["spec"],
    partial_variables={"format_instructions": feature_parser.get_format_instructions()},
)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

llm = ChatOllama(model="qwen3-vl:2b", temperature=0)

intent_chain = intent_prompt | llm
feature_chain = feature_prompt | llm


@app.route("/generate", methods=["POST"])
def handle_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get("input")

        if not user_prompt:
            return jsonify({"error": "Missing 'input' field"}), 400

        intent_response = intent_chain.invoke({"prompt": user_prompt})
        intent_data = intent_parser.parse(intent_response.content)

        print("Detected Intent:", intent_data)

        if intent_data["intent"] == "create_feature":
            feature_response = feature_chain.invoke({"spec": user_prompt})
            parsed_output = feature_parser.parse(feature_response.content)

            return jsonify(parsed_output), 200

        elif intent_data["intent"] == "modify_code":
            return jsonify({
                "message": "Modify code intent detected (session feature coming soon)."
            }), 200

        else:
            return jsonify({"error": "Unknown intent detected"}), 400

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "GenAI Flutter API is running ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)