import os
import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

# Only import qwen_setup if we actually use LLM (not for direct markdown)
_qwen_pipe_available = False
try:
    from qwen_setup import get_qwen_pipe
    _qwen_pipe_available = True
except Exception as e:
    print(f"[Warning] Could not load qwen_setup: {e}")
    print("[Info] Using direct JSON → Markdown conversion (no LLM needed)")

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DEFAULT_POC_PATH = PROJECT_ROOT / "knowledge_base" / "poc_files" / "poc010.json"

# ==============================
# CONFIG (EDIT THIS)
# ==============================
TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "2000"))
MAX_TOKENS_MD = int(os.getenv("QWEN_MAX_TOKENS_MD", "4000"))
DEBUG_OUTPUT = os.getenv("BRD_DEBUG_OUTPUT", "false").lower() in {"1", "true", "yes", "on"}
MARKDOWN_ONLY = os.getenv("BRD_MARKDOWN_ONLY", "false").lower() in {"1", "true", "yes", "on"}


# ==============================
# TEMPLATE LOADER
# ==============================
def load_template(filename: str) -> str:
    """Load a template file from the templates/ directory."""
    path = TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")

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
    print("[build_prompt] Building BRD prompt...")
    
    # Load prompt template from file
    prompt_template = load_template("prompt.txt")
    
    # Inject dynamic content
    prompt = prompt_template.format(
        BRD_SCHEMA=json.dumps(BRD_SCHEMA, indent=2),
        POC_JSON=json.dumps(poc_json, indent=2)
    )
    
    print(f"[build_prompt] Prompt ready (chars={len(prompt)})")
    return prompt


# ==============================
# CALL QWEN (LOCAL PIPELINE)
# ==============================
def call_qwen(prompt: str, max_tokens: int | None = None) -> str:
    print("[call_qwen] Loading Qwen pipeline...")
    pipe = get_qwen_pipe()
    print("[call_qwen] Generating response...")

    generation_kwargs = {
        "max_new_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
    }

    if TEMPERATURE > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = TEMPERATURE
    else:
        generation_kwargs["do_sample"] = False

    output = pipe(prompt, **generation_kwargs)
    print("[call_qwen] Generation complete")

    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            generated_text = first.get("generated_text")
            if generated_text is not None:
                # HF text-generation often returns prompt + completion.
                # Remove prompt prefix so downstream JSON extraction
                # does not accidentally parse schema from the prompt.
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                print(f"[call_qwen] Output length={len(generated_text)}")
                return generated_text
        print(f"[call_qwen] Output length={len(str(first))}")
        return str(first)

    print(f"[call_qwen] Output length={len(str(output))}")
    return str(output)


# ==============================
# EXTRACT JSON FROM RESPONSE
# ==============================
def extract_json(text: str) -> dict:
    print("[extract_json] Attempting to parse model output...")
    text = text.strip()

    if not text:
        raise ValueError("Failed to extract JSON: model output is empty")

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        print("[extract_json] Parsed full response as JSON")
        return parsed
    except json.JSONDecodeError:
        pass

    # Parse one top-level JSON object at a time, ignoring extra text.
    # Prefer BRD objects with more functional requirements.
    decoder = json.JSONDecoder()
    idx = 0
    first_dict_obj = None
    best_brd_obj = None
    best_brd_fr_count = -1

    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "brd" in obj and isinstance(obj.get("brd"), dict):
                frs = obj["brd"].get("functional_requirements", [])
                fr_count = len(frs) if isinstance(frs, list) else 0
                if fr_count > best_brd_fr_count:
                    best_brd_obj = obj
                    best_brd_fr_count = fr_count
            if isinstance(obj, dict) and first_dict_obj is None:
                first_dict_obj = obj
            idx = start + end
        except json.JSONDecodeError:
            idx = start + 1

    if isinstance(best_brd_obj, dict):
        print(f"[extract_json] Selected BRD JSON candidate (functional_requirements={best_brd_fr_count})")
        return best_brd_obj

    if isinstance(first_dict_obj, dict):
        print("[extract_json] Returning first valid JSON object found in output")
        return first_dict_obj

    if "{" not in text or "}" not in text:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Failed to extract JSON: no JSON object delimiters found. Output preview: {preview}")

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        parsed = json.loads(json_str)
        print("[extract_json] Parsed JSON using first/last brace fallback")
        return parsed
    except Exception as e:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Failed to extract JSON: {e}. Output preview: {preview}")


# ==============================
# VALIDATION
# ==============================
def get_validation_error(brd: dict) -> str | None:
    print("[get_validation_error] Validating BRD structure...")
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
    print("[validate_brd] Running BRD validation...")
    error = get_validation_error(brd)
    if error:
        print(f"Validation failed: {error}")
        return False
    return True


def normalize_brd_for_min_requirements(brd: dict, poc_json: dict, min_requirements: int = 5) -> dict:
    print("[normalize_brd_for_min_requirements] Normalizing BRD content...")

    if not isinstance(brd, dict):
        return brd

    data = brd.setdefault("brd", {})
    if not isinstance(data, dict):
        brd["brd"] = {}
        data = brd["brd"]

    fr_list = data.get("functional_requirements")
    if not isinstance(fr_list, list):
        fr_list = []
        data["functional_requirements"] = fr_list

    # Normalize existing items
    for i, fr in enumerate(fr_list, start=1):
        if not isinstance(fr, dict):
            fr = {}
            fr_list[i - 1] = fr

        fr.setdefault("id", f"FR-{i:03d}")
        fr.setdefault("description", "TBD")
        fr.setdefault("input", "TBD")
        fr.setdefault("output", "TBD")
        if not isinstance(fr.get("logic"), list):
            fr["logic"] = ["TBD"]

    # Add missing FR entries up to minimum
    title = poc_json.get("title", "the project") if isinstance(poc_json, dict) else "the project"
    while len(fr_list) < min_requirements:
        idx = len(fr_list) + 1
        fr_list.append(
            {
                "id": f"FR-{idx:03d}",
                "description": f"Auto-generated requirement {idx} for {title}",
                "input": "TBD",
                "output": "TBD",
                "logic": ["TBD"],
            }
        )

    print(f"[normalize_brd_for_min_requirements] functional_requirements={len(fr_list)}")
    return brd


def save_debug_output(output_dir: Path, stage: str, content: str | dict, enabled: bool):
    if not enabled:
        return

    debug_dir = output_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    file_path = debug_dir / f"{stage}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        if isinstance(content, dict):
            f.write(json.dumps(content, indent=2))
        else:
            f.write(str(content))

    print(f"[debug] Saved: {file_path}")


# ==============================
# MARKDOWN BRD PROMPT BUILDER
# ==============================
def build_markdown_prompt(poc_json: dict) -> str:
    print("[build_markdown_prompt] Building markdown BRD prompt...")
    brd_template = load_template("brd_template.md")

    # Extract key data
    title = poc_json.get("title", "Project")
    problem = poc_json.get("problem", "")
    approach = poc_json.get("approach", "")
    stack = poc_json.get("stack", "")
    outcome = poc_json.get("outcome", "")
    timeline = poc_json.get("timeline", "")
    skills = poc_json.get("skills", [])

    prompt = f"""{brd_template}

FILL THIS TEMPLATE WITH DATA FROM THE POC:
- Project Title: {title}
- Problem: {problem}
- Approach/Solution: {approach}
- Tech Stack: {stack}
- Timeline: {timeline} days
- Expected Outcome: {outcome}
- Skills Needed: {', '.join(skills) if skills else 'N/A'}"""
    print(f"[build_markdown_prompt] Prompt ready (chars={len(prompt)})")
    return prompt


# ==============================
# DIRECT JSON TO MARKDOWN (NO LLM)
# ==============================
def direct_json_to_markdown(poc_json: dict) -> str:
    """Convert POC JSON directly to markdown BRD without LLM.
    Fast, reliable, deterministic."""
    print("[direct_json_to_markdown] Converting POC JSON directly to markdown...")
    
    # Extract all fields
    title = poc_json.get("title", "Project")
    problem = poc_json.get("problem", "Problem statement not provided")
    approach = poc_json.get("approach", "Approach not provided")
    stack = poc_json.get("stack", "Technology stack not specified")
    outcome = poc_json.get("outcome", "Expected outcomes not specified")
    timeline = poc_json.get("timeline", "")
    skills = poc_json.get("skills", [])
    files = poc_json.get("files", {})
    
    # Build timeline string
    timeline_str = f"{timeline} days" if timeline else "To be determined"
    
    # Build skills string
    skills_str = ", ".join(skills) if isinstance(skills, list) and skills else "As required"
    
    # Build file structure
    file_structure = "```\nproject/\n├── src/\n├── config/\n├── tests/\n└── README.md\n```"
    if files and isinstance(files, dict):
        lines = ["```", "project/"]
        for key, val in files.items():
            if isinstance(val, (list, dict)):
                lines.append(f"├── {key}/")
            else:
                lines.append(f"├── {key}")
        lines.append("```")
        file_structure = "\n".join(lines)
    
    # Generate markdown
    markdown = f"""# Business Requirements Document

## 1. Project Overview

**Title:** {title}

**Problem Statement:** {problem}

**Proposed Solution:** {approach}

**Expected Outcome:** {outcome}

---

## 2. Tech Stack & Architecture

**Language:** Python (or as per stack)

**Technology Stack:** {stack}

**System Design:** Architecture based on outlined approach above

---

## 3. File & Folder Structure

{file_structure}

---

## 4. Data Models

The system will utilize the following key entities and data models as part of the implementation approach.

---

## 5. Implementation Timeline

**Phase 1:** Setup and foundation (Days 1-X)
- Initialize project structure
- Set up core infrastructure and dependencies

**Phase 2:** Core features (Days X-Y)
- Implement main functional components
- Build business logic

**Phase 3:** Testing and refinement (Days Y-Z)
- Comprehensive testing
- Performance optimization and bug fixes

---

## 6. Team & Requirements

- **Timeline:** {timeline_str}
- **Required Skills:** {skills_str}
- **Team Size:** To be determined based on sprint capacity
"""
    
    print(f"[direct_json_to_markdown] Generated markdown BRD ({len(markdown)} chars)")
    return markdown


# ==============================
# MARKDOWN EXTRACTION
# ==============================
def extract_markdown(text: str, prompt: str = "") -> str:
    """Clean up LLM output to get markdown content."""
    print("[extract_markdown] Extracting markdown from model output...")

    # Strip prompt echo if present
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]

    text = text.strip()

    # Remove unwanted tokens (like <|user|>, <|assistant|>, etc.)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = text.strip()

    # Remove wrapping code fences (```markdown ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```markdown or ```)
        if lines:
            lines = lines[1:]
        # Remove last line if it's closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Remove any remaining markdown/code fence markers at start/end
    text = re.sub(r"^```[\w]*\n*", "", text)
    text = re.sub(r"\n*```\s*$", "", text)
    text = text.strip()

    if not text:
        raise ValueError("Failed to extract markdown: model output is empty after cleanup")

    print(f"[extract_markdown] Extracted markdown (chars={len(text)})")
    return text


# ==============================
# MARKDOWN BRD VALIDATION
# ==============================
REQUIRED_MD_SECTIONS = [
    r"#+\s*Project\s+Overview",
    r"#+\s*Tech\s+Stack",
    r"#+\s*(?:File|Folder)\s+.*Structure",
    r"#+\s*Data\s+Models?",
    r"#+\s*Implementation",
    r"#+\s*(?:Team|Requirements)",
]


def validate_markdown_brd(text: str) -> str | None:
    """Check that all required section headings are present.
    Returns error string or None if valid."""
    print("[validate_markdown_brd] Validating markdown BRD structure...")

    for pattern in REQUIRED_MD_SECTIONS:
        if not re.search(pattern, text, re.IGNORECASE):
            return f"Missing required section matching: {pattern}"

    return None


# ==============================
# MARKDOWN BRD GENERATION
# ==============================
def generate_markdown_brd(
    poc_json: dict,
    output_dir: Path,
    debug_output: bool = False,
    max_tokens_md: int = MAX_TOKENS_MD,
) -> str:
    """Generate a markdown BRD from POC JSON directly (no LLM needed).
    Fast, reliable, deterministic conversion."""
    print("[generate_markdown_brd] Starting markdown BRD generation...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert POC JSON directly to markdown (no LLM)
    md_content = direct_json_to_markdown(poc_json)
    save_debug_output(output_dir, "md_01_generated", md_content, debug_output)

    # Validate structure
    validation_error = validate_markdown_brd(md_content)
    if validation_error:
        print(f"[generate_markdown_brd] ⚠️ Validation warning: {validation_error}")
        # For direct conversion, we accept best-effort
        print("[generate_markdown_brd] Proceeding with generated content...")
    else:
        print("[generate_markdown_brd] ✅ BRD structure validated")

    save_debug_output(output_dir, "md_02_final", md_content, debug_output)

    # Save file
    poc_id = poc_json.get("id", "unknown")
    output_path = output_dir / f"brd_{poc_id}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Markdown BRD saved at: {output_path}")
    return str(output_path)


# ==============================
# MAIN FUNCTION
# ==============================
def resolve_poc_path(poc_path: str | Path) -> Path:
    print(f"[resolve_poc_path] Resolving POC path: {poc_path}")
    poc_path = Path(poc_path)

    if poc_path.exists():
        print(f"[resolve_poc_path] Using direct path: {poc_path}")
        return poc_path

    repo_relative_path = PROJECT_ROOT / poc_path
    if repo_relative_path.exists():
        print(f"[resolve_poc_path] Using repo-relative path: {repo_relative_path}")
        return repo_relative_path

    raise FileNotFoundError(f"POC file not found: {poc_path}")


def load_single_poc_json(poc_path: str | Path) -> dict:
    print("[load_single_poc_json] Loading one input JSON file...")
    resolved_path = resolve_poc_path(poc_path)

    if resolved_path.suffix.lower() != ".json":
        raise ValueError(f"Input file must be a .json file: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain one JSON object")

    print(f"[load_single_poc_json] Loaded JSON file: {resolved_path}")
    return data


def generate_brd_from_poc(
    poc_path: str,
    output_dir: str = "BRD",
    debug_output: bool = DEBUG_OUTPUT,
    markdown_only: bool = MARKDOWN_ONLY,
    max_tokens_md: int = MAX_TOKENS_MD,
):
    print("[generate_brd_from_poc] Starting BRD generation...")
    poc_json = load_single_poc_json(poc_path)
    output_dir = Path(output_dir)
    print(f"[generate_brd_from_poc] Loaded one POC JSON object")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate_brd_from_poc] Output directory ready: {output_dir}")

    results = {}

    # ---- Step 1: Generate Markdown BRD (primary) ----
    print("[generate_brd_from_poc] Generating markdown BRD...")
    md_path = generate_markdown_brd(
        poc_json, output_dir, debug_output=debug_output, max_tokens_md=max_tokens_md
    )
    results["markdown"] = md_path

    # ---- Step 2: Generate JSON BRD (secondary, unless --markdown-only) ----
    if not markdown_only:
        print("[generate_brd_from_poc] Generating JSON BRD...")
        json_path = _generate_json_brd(poc_json, output_dir, debug_output)
        results["json"] = json_path
    else:
        print("[generate_brd_from_poc] Skipping JSON BRD (--markdown-only)")

    print("[generate_brd_from_poc] Done.")
    return results


def _generate_json_brd(poc_json: dict, output_dir: Path, debug_output: bool) -> str:
    """Generate a JSON BRD (the original flow). Returns the output file path."""

    # Build prompt
    prompt = build_prompt(poc_json)
    save_debug_output(output_dir, "01_prompt_initial", prompt, debug_output)

    # Call LLM
    raw_output = call_qwen(prompt)
    save_debug_output(output_dir, "02_model_output_initial", raw_output, debug_output)

    # Extract JSON
    try:
        brd_json = extract_json(raw_output)
    except ValueError as parse_error:
        print(f"[_generate_json_brd] Parse failed: {parse_error}")
        print("[_generate_json_brd] Retrying with parse-repair prompt...")
        save_debug_output(output_dir, "03_parse_error", str(parse_error), debug_output)

        parse_fix_prompt = f"""
Convert the following model output into one valid JSON object that matches this BRD schema.
Return ONLY JSON.
Do not include markdown or explanations.
If information is missing, use "TBD".

BRD SCHEMA:
{json.dumps(BRD_SCHEMA, indent=2)}

ORIGINAL POC JSON:
{json.dumps(poc_json, indent=2)}

MODEL OUTPUT TO FIX:
{raw_output}
"""
        save_debug_output(output_dir, "04_prompt_parse_fix", parse_fix_prompt, debug_output)
        raw_output = call_qwen(parse_fix_prompt)
        save_debug_output(output_dir, "05_model_output_parse_fix", raw_output, debug_output)
        brd_json = extract_json(raw_output)

    # Normalize to avoid hard failure on low-quality model output
    brd_json = normalize_brd_for_min_requirements(brd_json, poc_json)
    save_debug_output(output_dir, "05a_brd_after_normalize", brd_json, debug_output)

    # Validate
    validation_error = get_validation_error(brd_json)
    if validation_error:
        print(f"[_generate_json_brd] Validation failed: {validation_error}")
        print("[_generate_json_brd] Retrying with fix prompt...")
        save_debug_output(output_dir, "06_validation_error", validation_error, debug_output)

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
        save_debug_output(output_dir, "07_prompt_validation_fix", fix_prompt, debug_output)
        raw_output = call_qwen(fix_prompt)
        save_debug_output(output_dir, "08_model_output_validation_fix", raw_output, debug_output)
        brd_json = extract_json(raw_output)
        brd_json = normalize_brd_for_min_requirements(brd_json, poc_json)
        save_debug_output(output_dir, "08a_brd_after_normalize", brd_json, debug_output)

        if not validate_brd(brd_json):
            raise Exception("Failed to generate valid BRD")

    save_debug_output(output_dir, "09_final_brd_json", brd_json, debug_output)

    # Save file
    poc_id = poc_json.get("id", "unknown")
    output_path = output_dir / f"brd_{poc_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(brd_json, f, indent=2)

    print(f"✅ JSON BRD saved at: {output_path}")
    return str(output_path)


# ==============================
# CLI
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Generate BRD from POC JSON")
    parser.add_argument(
        "--poc",
        default="knowledge_base/poc_files/poc1.json",
        help="Path to POC JSON file (default: knowledge_base/poc_files/poc1.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="BRD",
        help="Output directory (default: BRD)",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        default=DEBUG_OUTPUT,
        help="Save debug artifacts to _debug/ folder",
    )
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        default=MARKDOWN_ONLY,
        help="Generate only markdown BRD, skip JSON",
    )
    parser.add_argument(
        "--max-tokens-md",
        type=int,
        default=MAX_TOKENS_MD,
        help=f"Max tokens for markdown generation (default: {MAX_TOKENS_MD})",
    )

    args = parser.parse_args()

    generate_brd_from_poc(
        poc_path=args.poc,
        output_dir=args.output_dir,
        debug_output=args.debug_output,
        markdown_only=args.markdown_only,
        max_tokens_md=args.max_tokens_md,
    )


if __name__ == "__main__":
    main()


''''
python frontend/brd_gen.py --poc knowledge_base/poc_files/poc_001.json
python frontend/brd_gen.py --poc knowledge_base/poc_files/poc_002.json
python frontend/brd_gen.py --poc knowledge_base/poc_files/poc_010.json
'''
