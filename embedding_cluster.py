import os
import ast
import autopep8
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from litellm import embedding
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_NIM_API_KEY", "")
os.environ["NVIDIA_NIM_API_BASE"] = os.getenv("NVIDIA_NIM_API_BASE", "")

def remove_comments_and_docstrings(source: str) -> str:
    """
    Remove Markdown fences, comments, and docstrings from Python source code.
    """

    # --- Step 1: Remove Markdown code fences like ``` and ```python
    source = re.sub(r"```(?:\w+)?", "", source).strip()

    # --- Step 2: Use AST to remove docstrings ---
    class RemoveDocstring(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)):
                node.body = node.body[1:]  # remove function docstring
            return node
        def visit_AsyncFunctionDef(self, node):  # handle async defs too
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)):
                node.body = node.body[1:]
            return node
        def visit_ClassDef(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)):
                node.body = node.body[1:]  # remove class docstring
            return node
        def visit_Module(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)):
                node.body = node.body[1:]  # remove module docstring
            return node

    try:
        tree = ast.parse(source)
        tree = RemoveDocstring().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
    except Exception:
        code = source  # fallback if AST fails

    # --- Step 3: Remove comments (full-line and inline) ---
    cleaned_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        # Skip pure comment lines
        if stripped.startswith("#"):
            continue
        # Remove inline comments (but not inside strings)
        line_no_inline = re.sub(r'(?<!["\'])#.*', '', line)
        cleaned_lines.append(line_no_inline.rstrip())

    return "\n".join(cleaned_lines).strip()

def standardize_code(code):
    """
    Standardize code to PEP8 using autopep8.
    """
    try:
        code_pep8 = autopep8.fix_code(code, options={'aggressive': 2})
        return code_pep8
    except Exception:
        return code

def get_nvidia_embedding(text, model, input_type="query"):
    response = embedding(
        model=model,
        input=[text],
        input_type=input_type
    )
    return np.array(response['data'][0]['embedding'])

def cluster_by_embedding(heuristic_files, alpha=0.95, model_name="nvidia_nim/nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"):
    codes = []
    file_names = []
    for hfile in heuristic_files:
        with open(hfile, "r", encoding="utf-8") as f:
            code = f.read()
        code_clean = remove_comments_and_docstrings(code)
        code_pep8 = standardize_code(code_clean)
        codes.append(code_pep8)
        file_names.append(hfile)

    embeddings = [get_nvidia_embedding(code, model=model_name, input_type="query") for code in codes]
    clusters = []
    assignments = {}

    for idx, emb in enumerate(embeddings):
        assigned = False
        for cidx, cluster in enumerate(clusters):
            # Check similarity with all members in the cluster
            sims = [cosine_similarity([emb], [embeddings[member]])[0][0] for member in cluster]
            if all(sim >= alpha for sim in sims):
                cluster.append(idx)
                assignments[file_names[idx]] = f"cluster_{cidx+1}"
                assigned = True
                break
        if not assigned:
            clusters.append([idx])
            assignments[file_names[idx]] = f"cluster_{len(clusters)}"

    # Print results
    for cidx, cluster in enumerate(clusters):
        print(f"Cluster {cidx+1}:")
        for idx in cluster:
            print(f"  {os.path.basename(file_names[idx])}")

    return assignments, clusters

if __name__ == "__main__":
    folder = "outputs/main/hsevo-qd_bpp_online_2025-08-28_22-45-05"
    heuristic_files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if (
            f.startswith("problem_iter") and
            "_response" in f and
            f.endswith(".txt") and
            not f.endswith("_stdout.txt") and
            not f.endswith("_prompt.txt")
        )
    ])
    assignments, clusters = cluster_by_embedding(heuristic_files, alpha=0.99)
    with open("embedding_cluster_assignments.json", "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)