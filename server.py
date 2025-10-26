from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# ✅ Intent classification schema
intent_schemas = [
    ResponseSchema(name="intent", description="Either 'create_feature' or 'modify_code'"),
    ResponseSchema(name="target_files", description="List of file paths mentioned in the prompt")
]

intent_parser = StructuredOutputParser.from_response_schemas(intent_schemas)

intent_prompt = PromptTemplate(
    template="""
Analyze this user request:

{prompt}

Classify the task as:
- create_feature → generate clean architecture feature
- modify_code → update existing code

Output valid JSON only:
{format_instructions}
""",
    input_variables=["prompt"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()},
)

# ✅ Feature generation output schema
feature_schemas = [
    ResponseSchema(
        name="files",
        description="List of files to create. Each element: { path, content }"
    )
]

feature_parser = StructuredOutputParser.from_response_schemas(feature_schemas)

feature_prompt = PromptTemplate(
    template="""
You are a Flutter clean architecture project generator.

Create new feature code based on this description:

{spec}

Output valid JSON only:
{format_instructions}
""",
    input_variables=["spec"],
    partial_variables={"format_instructions": feature_parser.get_format_instructions()},
)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

intent_chain = intent_prompt | llm
feature_chain = feature_prompt | llm


@app.route("/generate", methods=["POST"])
def handle_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get("feature_spec")

        if not user_prompt:
            return jsonify({"error": "feature_spec is required"}), 400

        # ✅ Step 1: Detect intent
        intent_res = intent_chain.invoke({"prompt": user_prompt})
        intent_data = intent_parser.parse(intent_res.content)

        print("Intent:", intent_data)

        # ✅ Create Feature Flow
        if intent_data["intent"] == "create_feature":
            response = feature_chain.invoke({"spec": user_prompt})
            raw_output = response.content
            parsed = feature_parser.parse(raw_output)
            return jsonify({"files": parsed["files"]}), 200

        # ✅ Modify Code Flow
        elif intent_data["intent"] == "modify_code":
            return jsonify({
                "message": "Code modification detected (🛠 Implementation coming soon)",
                "target_files": intent_data["target_files"]
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
