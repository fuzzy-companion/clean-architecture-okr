const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  console.log("Flutter Gen AI Extension activated!");

  let disposable = vscode.commands.registerCommand(
    "flutterGenAI.generate",
    async function () {
      try {
        // Step 1. Ask user for project idea
        const prompt = await vscode.window.showInputBox({
          prompt: "Describe your Flutter project or feature idea",
          placeHolder: "Example: Todo app with Firebase auth and clean architecture",
        });

        if (!prompt) {
          vscode.window.showWarningMessage("No input provided.");
          return;
        }

        vscode.window.showInformationMessage("Generating project structure...");

        // Step 2. Send prompt to your GenAI backend (e.g., LangChain/FastAPI)
        const res = await axios.post("http://localhost:8000/generate", {
            feature_spec: prompt,
        });

        const { files } = res.data; // Expecting { files: [{ path, content }] }

        // Step 3. Write files into current workspace
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
          vscode.window.showErrorMessage("Open a folder before generating files.");
          return;
        }

        const rootPath = workspaceFolders[0].uri.fsPath;

        files.forEach((file) => {
          const filePath = path.join(rootPath, file.path);
          fs.mkdirSync(path.dirname(filePath), { recursive: true });
          fs.writeFileSync(filePath, file.content, "utf8");
        });

        vscode.window.showInformationMessage("✅ Flutter clean architecture project created!");
      } catch (err) {
        console.error(err);
        vscode.window.showErrorMessage("Error generating project: " + err.message);
      }
    }
  );

  context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
