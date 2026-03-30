import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POC_PATH = PROJECT_ROOT / "knowledge_base" / "poc_files" / "poc1.json"

# ==============================
# CONFIG (EDIT THIS)
# ==============================
QWEN_API_URL = os.getenv("QWEN_API_URL", "http://<YOUR_JUMP_SERVER_ENDPOINT>/generate")
API_KEY = os.getenv("QWEN_API_KEY") or None

TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "2000"))
REQUEST_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "60"))

# ==============================
# BRD SCHEMA (STRICT CONTRACT)
# ==============================
BRD_SCHEMA = {
    "brd": {
        "project_overview": {
            "title": "",
            "description": "",
            "problem_statement": "",
            "expected_outcome": ""
        },
        "objectives": {
            "business_objectives": [],
            "technical_objectives": []
        },
        "system_design": {
            "architecture": "",
            "components": [],
            "integrations": []
        },
        "api_contracts": [],
        "data_model": {
            "entities": []
        },
        "functional_requirements": [],
        "technical_details": {
            "language": "",
            "stack": [],
            "deployment": "TBD"
        },
        "timeline": {
            "estimate": "",
            "milestones": []
        },
        "risks_and_dependencies": {
            "risks": [],
            "dependencies": []
        },
        "assumptions": []
    }
}

# ==============================
# PROMPT BUILDER
# ==============================
def build_prompt(poc_json: dict) -> str:
    return f"""
You are a software architect and business analyst AI.

Your job is to convert the input POC JSON into a STRICT, STRUCTURED, MACHINE-READABLE BRD.

STRICT RULES:
- Output ONLY valid JSON
- No explanations, no markdown
- Do NOT skip fields
- Do NOT add extra fields
- Use the exact top-level structure shown in BRD SCHEMA

QUALITY RULES:
- Avoid vague language
- Use explicit inputs, outputs, and logic
- Minimum 5 functional requirements
- IDs must be FR-001 format
- If data is missing, use "TBD" instead of inventing facts
- Put any reasonable inference into assumptions

- Each functional requirement must include:
  id, description, input, output, logic (array)

- API contracts must include:
  endpoint, method, request_schema, response_schema

- Data model must include:
    entities array

- Each entity must include:
    name, fields

- Each field must include:
    name, type, required

BRD SCHEMA:
{json.dumps(BRD_SCHEMA, indent=2)}

INPUT POC JSON:
{json.dumps(poc_json, indent=2)}
"""


# ==============================
# CALL QWEN API
# ==============================
def call_qwen(prompt: str) -> str:
    if not QWEN_API_URL or "<YOUR_JUMP_SERVER_ENDPOINT>" in QWEN_API_URL:
        raise ValueError("QWEN_API_URL is not configured. Set it in your environment or .env file.")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(
            QWEN_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise Exception(f"Qwen API request failed: {e}") from e

    if response.status_code != 200:
        raise Exception(f"Qwen API error: {response.text}")

    try:
        body = response.json()
    except ValueError:
        return response.text

    if isinstance(body, dict):
        return body.get("text") or body.get("response") or body.get("output") or json.dumps(body)

    return str(body)


# ==============================
# EXTRACT JSON FROM RESPONSE
# ==============================
def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to extract JSON: {e}")


# ==============================
# VALIDATION
# ==============================
def get_validation_error(brd: dict) -> str | None:
    if not isinstance(brd, dict):
        return "Output must be a JSON object"

    data = brd.get("brd")
    if not isinstance(data, dict):
        return "Missing 'brd' object"

    required_sections = [
        "project_overview",
        "objectives",
        "system_design",
        "api_contracts",
        "data_model",
        "functional_requirements",
        "technical_details",
        "timeline",
        "risks_and_dependencies",
        "assumptions",
    ]

    for key in required_sections:
        if key not in data:
            return f"Missing section: {key}"

    functional_requirements = data.get("functional_requirements")
    if not isinstance(functional_requirements, list) or len(functional_requirements) < 5:
        return "functional_requirements must contain at least 5 items"

    for index, fr in enumerate(functional_requirements, start=1):
        if not isinstance(fr, dict):
            return f"functional_requirements[{index}] must be an object"

        for key in ["id", "description", "input", "output", "logic"]:
            if key not in fr:
                return f"functional_requirements[{index}] missing '{key}'"

        if not str(fr["id"]).startswith("FR-"):
            return f"functional_requirements[{index}] id must start with 'FR-'"

        if not isinstance(fr["logic"], list):
            return f"functional_requirements[{index}] logic must be an array"

    api_contracts = data.get("api_contracts")
    if not isinstance(api_contracts, list):
        return "api_contracts must be an array"

    for index, api in enumerate(api_contracts, start=1):
        if not isinstance(api, dict):
            return f"api_contracts[{index}] must be an object"
        for key in ["endpoint", "method", "request_schema", "response_schema"]:
            if key not in api:
                return f"api_contracts[{index}] missing '{key}'"

    data_model = data.get("data_model")
    if not isinstance(data_model, dict) or "entities" not in data_model:
        return "data_model must contain an 'entities' array"

    return None


def validate_brd(brd: dict) -> bool:
    error = get_validation_error(brd)
    if error:
        print(f"Validation failed: {error}")
        return False
    return True


# ==============================
# MAIN FUNCTION
# ==============================
def resolve_poc_path(poc_path: str | Path) -> Path:
    poc_path = Path(poc_path)

    if poc_path.exists():
        return poc_path

    repo_relative_path = PROJECT_ROOT / poc_path
    if repo_relative_path.exists():
        return repo_relative_path

    raise FileNotFoundError(f"POC file not found: {poc_path}")


def generate_brd_from_poc(poc_path: str, output_dir: str = "BRD"):
    poc_path = resolve_poc_path(poc_path)
    output_dir = Path(output_dir)

    # Load POC
    with open(poc_path, "r", encoding="utf-8") as f:
        poc_json = json.load(f)

    # Build prompt
    prompt = build_prompt(poc_json)

    # Call LLM
    raw_output = call_qwen(prompt)

    # Extract JSON
    brd_json = extract_json(raw_output)

    # Validate
    validation_error = get_validation_error(brd_json)
    if validation_error:
        print("Retrying with fix prompt...")

        fix_prompt = f"""
Fix the following JSON to strictly match the schema.
Return only valid JSON.
Do not add extra top-level fields.
If information is missing, use "TBD".

VALIDATION ERROR:
{validation_error}

BRD SCHEMA:
{json.dumps(BRD_SCHEMA, indent=2)}

ORIGINAL POC JSON:
{json.dumps(poc_json, indent=2)}

BROKEN JSON:
{json.dumps(brd_json, indent=2)}
"""
        raw_output = call_qwen(fix_prompt)
        brd_json = extract_json(raw_output)

        if not validate_brd(brd_json):
            raise Exception("Failed to generate valid BRD")

    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    poc_id = poc_json.get("id", "unknown")
    output_path = output_dir / f"brd_{poc_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(brd_json, f, indent=2)

    print(f"✅ BRD saved at: {output_path}")

    return str(output_path)


# ==============================
# OPTIONAL CLI USAGE
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poc",
        default=str(DEFAULT_POC_PATH),
        help="Path to POC JSON file",
    )
    parser.add_argument("--out", default="BRD", help="Output folder")

    args = parser.parse_args()

    generate_brd_from_poc(args.poc, args.out)



    
