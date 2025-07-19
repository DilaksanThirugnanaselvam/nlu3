import streamlit as st
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

# === Load Model and Tokenizer ===
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/deepseek-r1-0528-qwen3-8b-unsloth-bnb-4bit",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/deepseek-r1-0528-qwen3-8b-unsloth-bnb-4bit")
    model = PeftModel.from_pretrained(base_model, "Dilaksan/NLA")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_model()

# === System Prompt ===
system_prompt = r"""
You are an expert in numerical linear algebra with deep knowledge.

Respond to each user question with a direct, well-structured answer using LaTeX for all mathematical notation.

Use LaTeX to represent:
- matrices (e.g., $A$),
- vectors (e.g., $\vec{x}$),
- norms (e.g., $\lVert x \rVert_2$),
- operators (e.g., $A^T A$, $\kappa(A)$),
- and complexity notations (e.g., $\mathcal{O}(n^3)$).

Do not include explanations about what you're doing.
Avoid meta-comments, apologies, or summaries.

Only provide the clean final answer using appropriate LaTeX syntax.
"""

# === Streamlit UI ===
st.set_page_config(page_title="Numerical Linear Algebra Bot", page_icon="🧮")
st.title("🧮 Numerical Linear Algebra Assistant")
st.markdown("Ask any **numerical linear algebra** question and get a LaTeX-formatted response.")

question = st.text_area("📌 Enter your question here:", height=150)

if st.button("Generate Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        full_prompt = system_prompt.strip() + "\n\nUser Question: " + question.strip()
        with st.spinner("Generating answer..."):
            try:
                response = pipe(full_prompt, max_new_tokens=1024)
                generated = response[0]["generated_text"]
                answer = generated.replace(full_prompt, "").strip()
                st.success("✅ Answer generated:")
                st.latex(answer)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
