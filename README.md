Documentation, product requirements table, and setup instructions in documenation.md.

Political chatbot intended to engage in meaningful discussion on political and legal subjects. Each user input is categorized by an LLM-as-judge classifier into one of 3 categories:

- Conversational transition ("Oh, ok.", "I understand.", etc.) -> routed directly to model.
- ResearchInquiry -> paraphrased and routed to Tavily -> returns a 3-part claim/evidence/reasoning schema with live sourcing.
- ConstitutionalInquiry -> sent to custom-built RAG pipeline on the US constitution using FAISS, OpenAI embeddings, and semantic chunking -> returns a 3-part claim/evidence/reasoning schema with a cited article or amendment.

Input queries are scored on both safety and relevance to ensure proper guardrails. Users have the option to expose model's internal reasoning, à la Claude. All responses come with in-text citations that are cross-referenceable by the user.
Frontend built with Gradio. Demo video included in the repo.
