from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Session memory --> TODO store it in database for retrival
sessions = {}


# intent detection schema, parser and prompt
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

Output STRICTLY valid JSON (escaped properly). Do not include code fences, markdown, or text outside JSON.
""",
    input_variables=["prompt"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()},
)

# feature generation schema, parser and prompt
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

# Creating a session and storing the conversation in the memory
def create_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "memory": ConversationBufferMemory(memory_key="feature_history", return_messages=True),
            "files": {}
        }
    return sessions[session_id]


@app.route("/generate", methods=["POST"])
def handle_prompt():
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        user_prompt = data.get("input")

        session = create_session(session_id=session_id)

        if not user_prompt:
            return jsonify({"error": "Missing 'input' field"}), 400

        intent_res = intent_chain.invoke({"prompt": user_prompt})
        intent_data = intent_parser.parse(intent_res.content)

        print("Detected Intent:", intent_data)

        if intent_data["intent"] == "create_feature":
            response = feature_chain.invoke({"spec": user_prompt,
                                             "feature_history": session["memory"].load_memory_variables({})
                                             })
            # Save generated files to session
            parsed_output = feature_parser.parse(response.content)
            
            for file in parsed_output["files"]:
                session["files"][file["path"]] = file["content"]
            
            session["memory"].save_context(
               {"input": user_prompt},
               {"output": "Feature generated"},
            )

            return jsonify(parsed_output), 200

        elif intent_data["intent"] == "modify_code":
            # Include context (previously generated files)
            context_files = "\n\n".join([
                f"// File: {key}\n{value}" for key, value in session["files"].items()
            ])

            prompt_with_context = f"""
                User wants to modify existing feature code.

                Existing files:
                {context_files}

                User instruction:
                {user_prompt}
                """
            response = feature_chain.invoke({"spec": prompt_with_context,
                                             "feature_history": session["memory"].load_memory_variables({})
                                             })
            parsed_output = feature_parser.parse(response.content)

            for f in parsed_output["files"]:
                session["files"][f["path"]] = f["content"]

            session["memory"].save_context({"input": user_prompt}, {"output": "Code modified."})
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