# Keural Nova — Client Brief (MKD)

Prepared for client inquiry. Keural Nova is MKD's Korean-specialized large language model,
developed, tuned, evaluated, and deployed in-house.

---

### 1. Applied model name, version, and provider

- **Product / solution provider:** **Keural Nova**, developed by **MKD**.
- **Versions:** v1.0 and v1.1 released; **v1.2 in active development** (next-generation, expanded
  Korean coverage and refined quality).
- **Architecture:** a modern **Mixture-of-Experts (MoE)** design — ~35B total parameters with
  only ~3B active per token, giving large-model quality at a fraction of the inference cost.
  Built on a leading open-source foundation model (Qwen 3.6, Apache 2.0 licensed); all Korean
  specialization, tuning, evaluation, and deployment engineering is MKD's own work.

### 2. Reason for selection and main features

- **Korean-first specialization:** tuned by MKD on a large, proprietary, commercially-clean
  Korean instruction dataset — producing natural, native-quality Korean rather than translated
  output.
- **Efficient MoE architecture:** delivers strong performance while activating only ~3B
  parameters per token — significantly lower serving cost than dense models of comparable
  quality.
- **Long context:** 256K-token context window for long documents and extended conversations.
- **Open, self-hostable foundation:** no per-token dependency on an external API vendor; the
  model can run entirely on client infrastructure.
- **Full-stack capability:** MKD owns the complete pipeline — Korean data curation, model
  training, benchmarking, and production serving.

### 3. Operation model

Flexible deployment to match the client's security and infrastructure needs:

- **On-premise / private cloud:** the model is self-hostable, so **all data stays within the
  client's environment** — ideal for enterprise, public-sector, and privacy-sensitive use.
- **API:** MKD can also provide a hosted API endpoint.
- **Hardware:** runs on standard datacenter GPUs (validated on NVIDIA H200-class hardware).
- No external per-token licensing lock-in.

### 4. Fine-tuning, continual pre-training (CPT), and RAG

- **Fine-tuning:** ✅ Applied. MKD performs large-scale supervised fine-tuning (SFT) using an
  efficient, MoE-aware adaptation method across the full model, on a curated Korean-majority
  instruction dataset (200K+ examples). This is the core of Keural's Korean specialization.
- **Continual pre-training (CPT):** available as an option on the roadmap for injecting
  client-specific domain knowledge; not required for the general Korean models.
- **RAG (Retrieval-Augmented Generation):** ✅ Supported. Keural is trained with Korean
  retrieval/grounded-answering data and integrates with client knowledge bases and document
  stores, so it can answer from the client's own up-to-date content.

### 5. Performance evaluation and test results

Evaluated with a standardized, reproducible benchmark harness (industry-standard Korean and
general suites), measured consistently against the foundation model:

- **Korean conversational understanding:** **+5.9 points on KoBEST** over the foundation model —
  a clear, measurable gain in the target domain.
- **Native-quality Korean output:** produces clean, natural Korean (no language-mixing artifacts
  seen in general foundation models).
- **General knowledge:** maintained at foundation-model level across Korean and English
  knowledge benchmarks (KMMLU, HAE-RAE, MMLU).
- **Structured / instruction-following responses:** markedly improved answer formatting and
  task-following.

*(Full per-benchmark tables available on request. Scope note: Keural is optimized for Korean
language understanding and conversation; it is not positioned as a code-generation model.)*

### 6. Licensing and scope for commercial/business use

- **Commercially usable:** ✅ The foundation model is released under the permissive **Apache 2.0**
  license, which explicitly permits commercial use, modification, and redistribution.
- **Clean training data:** MKD trains only on **commercially-clean licensed data**
  (MIT / Apache-2.0 / CC BY); non-commercial and restricted-license data is deliberately
  excluded.
- **Result:** the client can deploy Keural for commercial and business purposes, on their own
  infrastructure, with no per-token fees to an external model vendor. MKD provides the model,
  deployment, and support.

---

**In short:** Keural Nova is MKD's efficient, Korean-specialized, commercially-deployable
LLM — self-hostable, backed by a full in-house data-and-training pipeline, with measurable
Korean-language gains and a clean commercial licensing footing.
