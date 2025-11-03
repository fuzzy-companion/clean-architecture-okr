from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

intent_schemas = [
    ResponseSchema(name="intent", description="Either 'create_feature' or 'modify_code'"),
]

intent_parser = StructuredOutputParser.from_response_schemas(intent_schemas)

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
""",
    input_variables=["prompt"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()},
)

feature_schemas = [
    ResponseSchema(
        name="files",
        description="List of files to create. Each item: { path, content }"
    )
]

feature_parser = StructuredOutputParser.from_response_schemas(feature_schemas)

feature_prompt = PromptTemplate(
    template="""
You are an expert Flutter Clean Architecture code generator.

Task:
Generate a complete Flutter feature following Clean Architecture principles based on the user’s input specification.

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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

intent_chain = intent_prompt | llm
feature_chain = feature_prompt | llm


@app.route("/generate", methods=["POST"])
def handle_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get("input")

        if not user_prompt:
            return jsonify({"error": "Missing 'input' field"}), 400

        intent_res = intent_chain.invoke({"prompt": user_prompt})
        intent_data = intent_parser.parse(intent_res.content)

        print("Detected Intent:", intent_data)

        if intent_data["intent"] == "create_feature":
            response = feature_chain.invoke({"spec": user_prompt})
            parsed_output = feature_parser.parse(response.content)
            return jsonify(parsed_output), 200

        elif intent_data["intent"] == "modify_code":
            return jsonify({
                "message": "Modify code intent detected (update flow coming soon)."
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