# DSPy Optimizer Design Document

This document outlines the architecture for the DSPy Optimizer library, a framework for iteratively improving DSPy prompts.

## 1. High-Level Vision

The goal is to create a flexible, extensible, and user-friendly library that automates prompt engineering. The system will take a base prompt and a labeled dataset, and iteratively refine the prompt to maximize performance on the dataset, while avoiding regressions.

## 2. Core Architectural Pillars

The framework is built on a set of core principles to ensure it is robust and future-proof.

### 2.1. Pluggable Strategies
The core logic of the optimizer is broken down into distinct, swappable components. The user can choose the implementation for each component that best suits their needs. The primary pluggable strategies are:
- **Merger Strategy:** How a refined prompt suggestion is merged into the main prompt.
- **Validation Strategy:** How a new candidate prompt is validated to prevent regressions.
- **Scoring Strategy:** How a model's prediction is compared against a ground-truth label.

### 2.2. Registry and Auto-Discovery
To make using these strategies simple, the framework will include a central registry.
- **Decorator-based Registration:** Developers can add new strategies to the framework simply by adding a decorator (e.g., `@registry.validators.register("my_validator")`).
- **String-based Selection:** Users can select strategies by passing simple, memorable strings (e.g., `validation_strategy="full"`) to the main optimizer class.

### 2.3. Callback System for Auditing
To provide a detailed audit trail and allow for easy integration with tools like MLflow, the system will use a callback mechanism.
- The optimizer will call hooks at key points in the optimization process (e.g., `on_refinement_end`, `on_validation_success`).
- Users can provide a list of custom callback objects to log metrics, save artifacts, or even control the flow of the optimization (e.g., early stopping).

## 3. Detailed Component Design

### 3.1. Data Flow
- The standard data format throughout the library will be `dspy.Example`. This ensures compatibility with the DSPy ecosystem.
- The main `PromptOptimiser` will accept a `training_set` and a `validation_set`, both as `list[dspy.Example]`.

### 3.2. The Refiner
- The `Refiner` module is responsible for proposing improvements to the prompt.
- **Internal Logic:** It uses a sophisticated `dspy.Signature` that includes few-shot examples to guide the LLM. It is provided with rich context, including the original prompt, the failing example, the model's flawed reasoning, the incorrect prediction, the expected output, and a history of previously failed suggestions for the same example.
- **External Interface:** The `Refiner`'s final output is a `dspy.Prediction` object containing a structured patch (target block, operation, and content) ready for the `Merger`.

### 3.3. The Merger
- The `Merger` is responsible for applying a patch to the main prompt.
- The default implementation is a `BlockBasedMerger`, which deterministically applies the patch to the specified block in the prompt. This is safe and predictable.
- The pluggable strategy design allows for more complex, LLM-powered mergers to be added in the future.

### 3.4. The Validator
- The `Validator` is responsible for checking a new candidate prompt for regressions.
- The system ships with multiple strategies to allow users to manage the speed-vs-safety trade-off:
    1.  **`FullValidationStrategy` (Default):** The safest option. After every proposed merge, it runs the candidate prompt against a full, separate validation set.
    2.  **`BatchedTrainingSetValidationStrategy`:** A faster option. It validates a new prompt against the training set after a batch of `k` refinements.
    3.  **`SingleExampleValidationStrategy`:** The fastest option, ideal for rapid development. It validates a candidate prompt only against the single example that triggered the refinement.

### 3.5. The Scorer
- The scoring logic is fully user-defined to make the framework domain-agnostic.
- The user will provide a `scorer` function to the optimizer. This function takes a `dspy.Example` (with the gold label) and a `dspy.Prediction` and returns `True` or `False`.
- The interface is designed to be compatible with standard metrics from libraries like DSPy.

## 4. Directory Structure

The project follows a conventional Python library structure to enhance maintainability and usability.

```
dspy_optimizer/           # The main library source code
├── __init__.py           # Exposes public API
├── optimizer.py          # The main PromptOptimiser class
├── evaluator.py          # The Evaluator module
├── refiner.py            # The Refiner module
├── models.py             # Data models: PromptPatch, Config, etc.
└── strategies/           # A dedicated home for all pluggable strategies
    ├── __init__.py
    ├── registry.py     # The registry object and decorators
    ├── merger/
    │   ├── base.py       # MergerStrategy interface
    │   └── block_based.py
    ├── validation/
    │   ├── base.py       # ValidationStrategy interface
    │   ├── full.py
    │   ├── batched.py
    │   └── single_example.py
    └── scoring/
        ├── base.py       # Scorer function type definition
        └── common.py     # Common scorers (numeric, exact_match)

examples/                 # Top-level directory for examples
└── dutch_invoices/
    ├── optimize.py       # The script to run the invoice optimization
    ├── dataset.py        # Logic for loading and preparing the data
    └── data/             # The actual invoice data

notebooks/                # Top-level, as before
tests/                    # Top-level, as before
