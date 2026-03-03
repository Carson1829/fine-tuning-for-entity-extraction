MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
CHUNK_SIZE  = 1500
STRIDE      = 300
MAX_LENGTH  = 2048
OUTPUT_DIR  = "./lora-span-model"
DEVICE      = "cuda"

# LoRA
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05
LORA_MODULES     = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
TRAIN_EPOCHS     = 5
LEARNING_RATE    = 5e-5
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO     = 0.05

VALID_TAGS = ["definition", "theorem", "proof", "example", "name", "reference"]

FEW_SHOT_SYSTEM_PROMPT = (
    "You are an expert mathematical data extraction assistant. "
    "Extract structural spans from mathematical text. "
    "Valid tags:\n"
    "- definition: defines a new concept or object.\n"
    "- theorem: makes a rigorous, provable claim.\n"
    "- proof: proves or sketches a proof of a theorem.\n"
    "- example: illustrates a definition or theorem.\n"
    "- name: the name of a newly defined object or theorem.\n"
    "- reference: a reference to a previously defined name.\n\n"
    "Return ONLY a JSON array. Each element has keys 'tag' and 'text'. "
    "The 'text' value must be copied VERBATIM from the input. "
    "If no spans are found, return an empty array []."
)

SYSTEM_PROMPT = (
    "You are an expert mathematical data extraction assistant. "
    "Extract structural spans from mathematical text. "
    "Valid tags:\n"
    "- definition: defines a new concept or object.\n"
    "- theorem: makes a rigorous, provable claim.\n"
    "- proof: proves or sketches a proof of a theorem.\n"
    "- example: illustrates a definition or theorem.\n"
    "- name: the name of a newly defined object or theorem.\n"
    "- reference: a reference to a previously defined name.\n\n"
    "Return ONLY a JSON array. Each element has keys 'tag' and 'text'. "
    "The 'text' value must be copied VERBATIM from the input."
)

USER_PROMPT_TEMPLATE = (
    "Extract all labeled spans from the following text. "
    "Return a JSON array of objects with keys 'tag' and 'text'.\n\n"
    "Input:\n{chunk_text}"
)