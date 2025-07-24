# DSPy Prompt-Optimizer Framework

A flexible framework that **iteratively improves prompts**-via self-critique and self-repair—until a large-language model (LLM) achieves the desired output on a labelled dataset.
Although the reference implementation targets “amount-excluding-tax” extraction from Dutch invoices, the design is **domain-agnostic**: swap in a new *signature* (I/O schema) and a *base prompt*, and the optimiser will refine prompts for *any* task—information extraction, classification, transformation, or reasoning.

---

## 1. Why this project exists

| Problem                                                     | Traditional fix                               | Limitations                                       | This framework’s answer                                                                                              |
| ----------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Prompt engineering is manual, brittle, expensive.           | Humans iterate on prompts by trial-and-error. | Slow, non-repeatable, hard to audit.              | **Automatic prompt search** guided by test data and LLM self-reflection.                                             |
| Each business domain needs slightly different instructions. | Maintain ad-hoc prompt variants.              | Prompt zoo quickly diverges; hard to reuse ideas. | **Block-based prompt merging** collects successful heuristics/examples into one canonical prompt.                    |
| Prompts can regress when new edge-cases show up.            | Manual regression testing.                    | Easy to forget a hidden corner case.              | **Built-in regression loop** reruns frozen validation set after every merge; unsafe patches roll back automatically. |

---

## 2. Key ideas

1. **Evaluator → Refiner → Merger loop**
   *Evaluator* runs the current prompt; if output ≠ gold, *Refiner* asks the LLM to propose a patch that would fix it; *Merger* folds that patch into the universal prompt.
   Loops ≤ *k* times per sample to avoid infinite churn.
2. **Self-critique with history**
   Each refinement step receives **all previous failed prompts** so the LLM does not repeat old mistakes.
3. **Numerical tolerance / custom scorers**
   Scoring is pluggable: exact-match, fuzzy string, semantic similarity, numeric ±ϵ, etc.
4. **Block-based prompts**
   Prompt text is split into `### Task`, `### Output format`, `### Examples`, `### Heuristics`.
   The merger appends new examples or heuristics to the correct block—prompt stays readable and under token limits.
5. **Parallel evaluation, serial merging**
   Invoices can be processed in parallel threads, but merging is serial to avoid merge conflicts.
6. **Audit trail**
   Every prompt, decision, score and timestamp can be logged to SQLite/W\&B so you can answer “*Why did the prompt change on 2025-07-14 15:03?*”

---

## 3. Architecture at a glance

```text
┌─────────────┐     wrong?    ┌───────────────┐  prompt patch  ┌─────────────┐
│  Evaluator  ├──────────────►│    Refiner    ├───────────────►│    Merger   │
│  (LLM run)  │ yes           │  (LLM self-   │                │   (diff /   │
└─────┬───────┘               │  critique)    │                │ block merge)│
      │ no                    └──────┬────────┘                └──────┬──────┘
      └──────────────────────────────┴────────────────────────────────┘
                                   updated prompt
```

---

## 4. Directory layout

```
dspy_optimizer/
├── invoice_amount_optimizer.py      # Reference implementation
├── core/
│   ├── optimizer.py                 # Generic Optimiser class
│   ├── evaluator.py                 # Evaluator module
│   ├── refiner.py                   # Refiner module
│   ├── merger.py                    # Merger module
│   └── scoring.py                   # Pluggable scorers
├── examples/
│   └── dutch_invoices/              # Example dataset & scripts
├── notebooks/
│   └── develop.ipynb                # Example notebooks for development
├── data/                            # Example datasets
│   └── invoices/                    # Example invoices dataset
├── tests/                           # Unit & integration tests
└── README.md                        # ← you are here
```

---

## 5. Coding rules & standards

| Area                      | Standard                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| **Python version**        | ≥ 3.12 (pattern-matching, `typing` improvements).                                               |
| **Style**                 | PEP 8 + *black* (100 char line length) + *isort*.                                               |
| **Type hints**            | Mandatory in all public functions/classes. Use `mypy --strict`.                                 |
| **Docstrings**            | Google style. Public APIs must include *Args*, *Returns*, *Raises*, *Example*.                  |
| **Naming**                | `PascalCase` for classes, `snake_case` for functions/vars, ALL\_CAPS for constants.             |
| **Immutability**          | Prefer `dataclass(frozen=True)` for config objects.                                             |
| **Dependency management** | Use **uv**; keep `pyproject.toml` single-source of truth.                                   |
| **Logging**               | Standard `logging` lib. No `print` except in CLI demos. Use structured JSON if writing to file. |
| **Testing**               | **pytest** with 90 % line coverage target. Mock LLM calls using fixtures.                       |
| **CI**                    | GitHub Actions: lint → type-check → tests → build wheel.                                        |
| **Versioning**            | SemVer.                                                                                         |
| **Security**              | Secrets via env vars; *never* hard-code keys in repo.                                           |
| **Docs**                  | Use MkDocs Material; auto-generate API docs with `mkdocstrings`.                                |

---

## 6. How to port to **any** extraction/classification task

1. **Define a new DSPy `Signature`** describing your *input fields* and *output fields*.

   ```python
   class PhoneExtractor(dspy.Signature):
       file: Attachments = dspy.InputField()
       phone_number: str = dspy.OutputField()
   ```
2. **Write a seed prompt** inside a base class (or via cfg).

   ```python
   base_prompt = """
   ### Task
   Extract the Dutch phone number of the billing contact from the document...
   """
   ```
3. **Plug both into the generic `Optimiser`**.

   ```python
   opt = GenericPromptOptimiser(model, base_prompt, PhoneExtractor)
   opt.optimise(dataset)
   ```

The optimiser will reuse the same evaluator/refiner/merger logic; only the I/O schema and scoring function change.

---



### Happy automating your prompt engineering 🚀
