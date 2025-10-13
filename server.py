from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os, json

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

app = Flask(__name__)

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

# Step 3: Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
chain = prompt | llm


@app.route("/generate", methods=["POST"])
def generate_feature():
    try:
        # Expect JSON input like: { "feature_spec": { ... } }
        data = request.get_json()
        if not data or "feature_spec" not in data:
            return jsonify({"error": "Missing 'feature_spec' in request body"}), 400

        feature_spec = data["feature_spec"]

        # Step 4: Run chain
        response = chain.invoke({"feature_spec": json.dumps(feature_spec)})

        # Extract text content
        raw_output = response.content if hasattr(response, "content") else response

        # Parse JSON structure
        parsed = parser.parse(raw_output)

        return jsonify({"files": parsed["files"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "GenAI Flutter API is running 🚀"})


if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)
