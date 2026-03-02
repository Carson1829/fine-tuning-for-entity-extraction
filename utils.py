import re
import json
import unicodedata
import config


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + config.CHUNK_SIZE, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += config.CHUNK_SIZE - config.STRIDE
    return chunks


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"  JSON PARSE FAILED: {e}")
            print(f"  Attempted: {match.group()[:200]}")
            return []
    print("  NO JSON ARRAY FOUND IN OUTPUT")
    return []


def find_span_in_chunk(span_text, chunk_text_):
    # 1. Exact match
    idx = chunk_text_.find(span_text)
    if idx != -1:
        return idx, span_text

    # 2. Strip whitespace
    stripped = span_text.strip()
    idx = chunk_text_.find(stripped)
    if idx != -1:
        return idx, stripped

    # 3. First 80 chars
    prefix = stripped[:80]
    if len(prefix) > 20:
        idx = chunk_text_.find(prefix)
        if idx != -1:
            return idx, prefix

    # 4. Normalize whitespace and unicode
    def normalize(s):
        s = re.sub(r'\s+', ' ', s)
        return unicodedata.normalize('NFKC', s).strip()

    idx = normalize(chunk_text_).find(normalize(span_text))
    if idx != -1:
        return idx, span_text

    return -1, span_text

FEW_SHOT_EXAMPLES = [
    {
        "input": "**3.8 Theorem** (Existence).: _Given any abelian group \\(G\\), there exists a homology theory with coefficient group \\(G\\)._",
        "output": json.dumps([
            {"tag": "theorem", "text": "**3.8 Theorem** (Existence).: _Given any abelian group \\(G\\), there exists a homology theory with coefficient group \\(G\\)._"},
            {"tag": "name",    "text": "3.8 Theorem"}
        ], indent=2)
    },
    {
        "input": "**Definition 1.8:** Let \\(D\\) be a connection in a vector bundle \\(E\\longrightarrow X\\). Then the _curvature_ \\(\\mathbf{\\Theta}_{E}(D)\\) is defined to be that element \\(\\mathbf{\\Theta}\\in\\mathcal{E}^{2}(X,\\operatorname{Hom}(E,E))\\) such that the \\(\\mathbf{C}\\)-linear mapping has a certain representation.",
        "output": json.dumps([
            {"tag": "definition", "text": "**Definition 1.8:** Let \\(D\\) be a connection in a vector bundle \\(E\\longrightarrow X\\). Then the _curvature_ \\(\\mathbf{\\Theta}_{E}(D)\\) is defined to be that element \\(\\mathbf{\\Theta}\\in\\mathcal{E}^{2}(X,\\operatorname{Hom}(E,E))\\) such that the \\(\\mathbf{C}\\)-linear mapping has a certain representation."},
            {"tag": "name",       "text": "Definition 1.8"}
        ], indent=2)
    },
    {
        "input": "_Proof._ Let \\(\\mathcal{H}_{n}(X,A)=H_{n}(X,A;G)\\), singular homology with coefficients in \\(G\\). Then each of the axioms has been proved previously.",
        "output": json.dumps([
            {"tag": "proof", "text": "_Proof._ Let \\(\\mathcal{H}_{n}(X,A)=H_{n}(X,A;G)\\), singular homology with coefficients in \\(G\\). Then each of the axioms has been proved previously."}
        ], indent=2)
    },
    {
        "input": "Two elements \\(a\\), \\(b\\) of a group \\(G\\) are said to be _conjugate_ if \\(b=x^{-1}ax\\) for some \\(x\\in G\\). It is easy to see that conjugacy is an equivalence relation.",
        "output": json.dumps([
            {"tag": "definition", "text": "Two elements \\(a\\), \\(b\\) of a group \\(G\\) are said to be _conjugate_ if \\(b=x^{-1}ax\\) for some \\(x\\in G\\). It is easy to see that conjugacy is an equivalence relation."},
            {"tag": "name",       "text": "conjugate"}
        ], indent=2)
    },
    {
        "input": "Here's some unrelated text about the structure of the document. No mathematical entities are defined here.",
        "output": json.dumps([], indent=2)
    }
]

def build_messages_fs(chunk_text_):
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": (
                "Extract all labeled spans from the following text. "
                "Return a JSON array of objects with keys 'tag' and 'text'.\n\n"
                f"Input:\n{ex['input']}"
            )
        })
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({
        "role": "user",
        "content": (
            "Extract all labeled spans from the following text. "
            "Return a JSON array of objects with keys 'tag' and 'text'.\n\n"
            f"Input:\n{chunk_text_}"
        )
    })
    return messages

def build_messages_ft(chunk_text_):
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": config.USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text_)
        }
    ]