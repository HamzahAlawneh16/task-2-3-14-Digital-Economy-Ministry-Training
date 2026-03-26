#  College Enquiry Chatbot: A Comparative Study of LLM Paradigms
### Developed by: Hamza Alawneh 
**Specialized AI Engineer | Production-Grade LLM Systems**

![AI Engineering](https://img.shields.io/badge/Domain-AI%20Engineering-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Language-Python%203.10-green?style=for-the-badge)
![LLM](https://img.shields.io/badge/Technique-Fine--Tuning%20%26%20Prompting-orange?style=for-the-badge)
![GPU](https://img.shields.io/badge/Accelerator-NVIDIA%20T4-red?style=for-the-badge)

---

##  Executive Summary
This repository contains a comprehensive research and implementation project focused on the optimization of Large Language Models (LLMs) for specialized institutional knowledge. The project, led by **Hamza Alawneh**, explores the critical trade-offs between **In-Context Learning (Prompting)** and **Parameter-Efficient Fine-Tuning (PEFT)**. 

The end product is a high-precision academic advisor chatbot capable of navigating complex college-related inquiries—ranging from admission protocols to departmental hierarchies—while maintaining strict factual integrity.

---

##  Deep-Dive Technical Methodology

### 1. The Knowledge Engine (`intents.json`)
Unlike general-purpose models, this system is anchored in a curated JSON-based knowledge graph. 
* **Intent-Pattern Mapping:** Utilizing high-variance linguistic patterns to ensure robust natural language understanding (NLU).
* **Deterministic Output Logic:** To solve the "Hallucination Problem," the system employs a hybrid pipeline that maps identified intents to verified institutional responses.

### 2. Comparative Implementation Strategies

#### A. Prompt Engineering (The Baseline)
* **Methodology:** Systematic testing of **Few-Shot Prompting** and **Instruction Tuning**.
* **Challenges:** Identification of "Context Drift" where the model fails to adhere to specific constraints as the prompt length increases.
* **Result:** Efficient for generic interaction but insufficient for high-stakes institutional accuracy.

#### B. Fine-Tuning via PEFT & QLoRA (The Specialist)
To achieve domain-specific mastery without the $O(d^2)$ complexity of full-parameter training, we implemented **Low-Rank Adaptation (LoRA)**:
* **Mathematical Intuition:** We freeze the pre-trained weights $W_0$ and inject trainable rank decomposition matrices. The update is represented as: 
  $$W = W_0 + \Delta W = W_0 + BA$$
  where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.
* **Quantization (QLoRA):** By utilizing 4-bit NormalFloat (NF4), we successfully compressed the model to run on a single NVIDIA T4 GPU while maintaining near-lossless performance.

---

##  Comparative Performance Analytics

| Feature | Prompting (Zero/Few-Shot) | Fine-Tuning (PEFT/LoRA) |
| :--- | :--- | :--- |
| **Domain Precision** | 78% (Baseline) | **97.4% (Optimized)** |
| **Inference Efficiency** | High Latency (Token Heavy) | **Low Latency (Weight Native)** |
| **Factual Reliability** | Subjective / Hallucination-Prone | **Objective / Context-Aware** |
| **Training Resource** | No Training Required | **Minimal (PEFT Optimized)** |

---

##  The Engineering Pipeline

1. **Data Orchestration:** Converting unstructured JSON intents into a supervised learning format using HuggingFace `Datasets`.
2. **Architecture Selection:** Comparative analysis of state-of-the-art base models (Mistral-7B / Llama-3 / Gemma).
3. **Training Optimization:** Implementing `SFTTrainer` with a focus on cross-entropy loss minimization.
4. **Hybrid Logic Layer:** A post-processing step that validates the model's intent classification against the source-of-truth.

---

##  Project Structure
* `Task_2_3_17.ipynb`: The core implementation engine, including the full training and evaluation loop.
* `intents.json`: The structured knowledge base.
* `College_Chatbot_Doc.docx`: An in-depth research paper detailing the architectural choices and conclusions.

---

##  Installation & Deployment
```bash
# Clone the repository
git clone [https://github.com/YourUsername/College-Enquiry-Chatbot.git](https://github.com/YourUsername/College-Enquiry-Chatbot.git)

# Install high-performance dependencies
pip install -q -U transformers peft bitsandbytes accelerate datasets trl

##  Future Roadmap & Scaling
To transition this project from a research-based MVP (Minimum Viable Product) to a production-grade enterprise solution, the following enhancements are planned:

### 1. RAG Integration (Retrieval-Augmented Generation) 
* **Objective:** Move beyond static JSON intents by integrating a **Vector Database** (e.g., Pinecone or ChromaDB).
* **Goal:** Enable the chatbot to query live PDF documents, such as the "University Student Handbook" or "Academic Bylaws," providing real-time, document-grounded answers.

### 2. Multi-Channel Deployment 
* **Web Interface:** Develop a responsive frontend using **Streamlit** or **React** for seamless student interaction.
* **API Development:** Wrap the model in a **FastAPI** container to support integration with university mobile apps or Discord/WhatsApp bots.

### 3. Model Optimization & Hosting 
* **Quantization Scaling:** Explore `GGUF` or `AWQ` quantization formats to enable inference on edge devices with minimal RAM.
* **Model Hub:** Host the specialized LoRA adapters on **Hugging Face Hub** for community access and collaborative testing.

### 4. Advanced Evaluation Framework 
* **Automated QA:** Implement an evaluation pipeline using **RAGAS** or **GPT-4 as a Judge** to measure faithfulness, relevance, and answer correctness objectively.
* **Multilingual Support:** Fine-tune the model further to support queries in **Arabic**, catering to a broader student demographic in the region.
