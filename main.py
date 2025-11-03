import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#
# -------- LOAD LOCAL QWEN 72B GPTQ MODEL --------
#

print("Loading Qwen 2.5 72B GPTQ model...")
model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ Model loaded!")


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []

    # System instruction (optional)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Past conversation messages
    for msg in history_messages:
        messages.append(msg)

    # User prompt
    messages.append({"role": "user", "content": prompt})

    # Convert to Qwen chat format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7
    )

    out_ids = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(out_ids, skip_special_tokens=True)
    return response


#
# -------- MAIN RAG PIPELINE --------
#

async def main():

    # RAGAnything config
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Load local embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded!")

    async def local_embed(texts):
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: embed_model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings

    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=local_embed,
    )

    # Initialize RAG
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=None,   # Remove vision for now
    )

    # Process a PDF
    await rag.process_document_complete(
        file_path="./27pg.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # Query it
    response = await rag.aquery(
        "In ImageNet classification, what is the Model Source of ResNet50 Model?",
        mode="hybrid"
    )

    print("\n🧠 Response:\n", response)


if __name__ == "__main__":
    asyncio.run(main())
