import re
import json
import unicodedata
import config


def chunk_text(text):
    """Split a document into overlapping chunks of fixed character length.
    Stride overlap ensures spans near chunk boundaries appear fully in at least one chunk.
    Returns a list of (chunk_start, chunk_end, chunk_text) tuples."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + config.CHUNK_SIZE, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += config.CHUNK_SIZE - config.STRIDE  # advance by CHUNK_SIZE - STRIDE to create overlap
    return chunks


def extract_json(text):
    """Parse a JSON array from raw model output.
    Strips markdown code fences the model may wrap its output in,
    then extracts the first [...] array and parses it.
    Returns an empty list if parsing fails."""
    text  = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)  # re.DOTALL allows [...] to span multiple lines
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"JSON PARSE FAILED: {e}")
            print(f"Attempted: {match.group()[:200]}")
            return []
    print("NO JSON ARRAY FOUND IN OUTPUT")
    return []


def find_span_in_chunk(span_text, chunk_text_):
    """Find the character offset of a predicted span text within the chunk.
    Uses exact string matching since the model is prompted to copy verbatim.
    Returns (local_offset, matched_text) or (-1, span_text) if not found."""
    idx = chunk_text_.find(span_text)
    if idx != -1:
        return idx, span_text
    return -1, span_text


def build_messages_fs(chunk_text_):
    """Build the prompt messages for few-shot inference (no LoRA).
    Includes two hardcoded examples covering definition+theorem+proof+name
    and implicit definition+name to demonstrate the expected output format.
    The examples use synthetic math text to avoid contamination with val/test files."""
    messages = [
        {"role": "system", "content": config.FEW_SHOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract all labeled spans from the following text. "
                "Return a JSON array of objects with keys 'tag' and 'text'.\n"
                "Input: **Definition (3.1)**.: Let \\(R\\) be a ring. Its (Jacobson) **radical**\\(\\operatorname{rad}(R)\\) is defined to be the intersection of all its maximal ideals.\n\n**Proposition (3.2)**.: _Let \\(R\\) be a ring, \\(\\mathfrak{a}\\) an ideal, \\(x\\in R\\), and \\(u\\in R^{\\times}\\). Then \\(x\\in\\operatorname{rad}(R)\\) if and only if \\(u-xy\\in R^{\\times}\\) for all \\(y\\in R\\)._\n\n**Proof:** Assume \\(x\\in\\operatorname{rad}(R)\\). Given a maximal ideal \\(\\mathfrak{m}\\), suppose \\(u-xy\\in\\mathfrak{m}\\). Since \\(x\\in\\mathfrak{m}\\) too, also \\(u\\in\\mathfrak{m}\\), a contradiction. Thus \\(u-xy\\) is a unit by **(2.22)**.\n"
                "Output: [{\"tag\": \"definition\", \"text\": \"**Definition (3.1)**.: Let \\\\(R\\\\) be a ring. Its (Jacobson) **radical**\\\\(\\\\operatorname{rad}(R)\\\\) is defined to be the intersection of all its maximal ideals.\"}, {\"tag\": \"name\", \"text\": \"radical\"}, {\"tag\": \"theorem\", \"text\": \"**Proposition (3.2)**.: _Let \\\\(R\\\\) be a ring, \\\\(\\\\mathfrak{a}\\\\) an ideal, \\\\(x\\\\in R\\\\), and \\\\(u\\\\in R^{\\\\times}\\\\). Then \\\\(x\\\\in\\\\operatorname{rad}(R)\\\\) if and only if \\\\(u-xy\\\\in R^{\\\\times}\\\\) for all \\\\(y\\\\in R\\\\)._\"}, {\"tag\": \"name\", \"text\": \"Proposition (3.2)\"}, {\"tag\": \"proof\", \"text\": \"**Proof:** Assume \\\\(x\\\\in\\\\operatorname{rad}(R)\\\\). Given a maximal ideal \\\\(\\\\mathfrak{m}\\\\), suppose \\\\(u-xy\\\\in\\\\mathfrak{m}\\\\). Since \\\\(x\\\\in\\\\mathfrak{m}\\\\) too, also \\\\(u\\\\in\\\\mathfrak{m}\\\\), a contradiction. Thus \\\\(u-xy\\\\) is a unit by **(2.22)**.\"}]\n"
                "---\n"
                "Input: For any element \\(a\\) of a group \\(G\\), the set \\(N_{a}\\) of all elements of \\(G\\) which commute with \\(a\\), \\(N_{a}=\\{x\\in G\\colon xa=ax\\}\\), is closed under multiplication and inversion. Thus \\(N_{a}\\) is a subgroup of \\(G\\), called the _centralizer_ of \\(a\\) in \\(G\\).\n"
                "Output: [{\"tag\": \"definition\", \"text\": \"For any element \\\\(a\\\\) of a group \\\\(G\\\\), the set \\\\(N_{a}\\\\) of all elements of \\\\(G\\\\) which commute with \\\\(a\\\\), \\\\(N_{a}=\\\\{x\\\\in G\\\\colon xa=ax\\\\}\\\\), is closed under multiplication and inversion. Thus \\\\(N_{a}\\\\) is a subgroup of \\\\(G\\\\), called the _centralizer_ of \\\\(a\\\\) in \\\\(G\\\\).\"}, {\"tag\": \"name\", \"text\": \"centralizer\"}]\n"
                "---\n"
                f"Input: {chunk_text_}\n"
                "Output:"
            )
        }
    ]
    return messages


def build_messages_ft(chunk_text_):
    """Build the prompt messages for fine-tuned inference.
    Uses zero-shot prompt only — few-shot examples are unnecessary
    after training and would consume context window space."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": config.USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text_)
        }
    ]
    return messages