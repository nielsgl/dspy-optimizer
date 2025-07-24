import concurrent.futures as cf
from dataclasses import dataclass
import json
import math
import os
import re

from attachments import config
from attachments.dspy import Attachments
from dotenv import load_dotenv
import dspy
import pandas as pd
import pdfplumber
from tqdm import tqdm

from dspy_optimizer.utils import get_root

load_dotenv()

lm = dspy.LM(
    "azure/gpt-4o-mini",
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)


def accuracy(y_true: dspy.Prediction, y_pred: dspy.Prediction, trace=None):
    return y_true.amount == y_pred.amount


def make_example(item) -> dspy.Example:
    with pdfplumber.open(item["file_path"]) as pdf:
        total_pages = len(pdf.pages)

    pages = "" if total_pages <= 5 else "[pages:1,2,3,-2,-1]"

    data = {
        "file": Attachments(item["file_path"] + "[tile:1x1]" + pages),
        "amount": item["totaal_excl_btw"],
    }
    return dspy.Example(**data).with_inputs("file")


def make_example_tuple(item) -> tuple[Attachments, float]:
    with pdfplumber.open(item["file_path"]) as pdf:
        total_pages = len(pdf.pages)

    pages = "" if total_pages <= 5 else "[pages:1,2,3,-2,-1]"

    return Attachments(item["file_path"] + "[tile:1x1]" + pages), item["totaal_excl_btw"]


def make_dataset(data: pd.DataFrame) -> list[dspy.Example]:
    samples = []

    config.set_verbose(False)
    for item in data.iterrows():
        samples.append(make_example(item[1]))

    config.set_verbose(True)

    return samples


def make_dataset_tuple(data: pd.DataFrame) -> list[tuple[Attachments, float]]:
    samples = []

    config.set_verbose(False)
    for item in tqdm(data.iterrows(), total=data.shape[0]):
        samples.append(make_example_tuple(item[1]))

    config.set_verbose(True)

    return samples


def initial_eval(dataset_):
    basic_predictor = dspy.ChainOfThought(AmountExtractor)

    for idx, item in enumerate(dataset_):
        pred = basic_predictor(**item.inputs())

        if item.amount != pred.amount:
            print(f"Misclassified id {idx}, true: {item.amount},  \t pred: {pred.amount}")


def initial_eval_tuples(dataset_: list[tuple[Attachments, float]]):
    basic_predictor = dspy.ChainOfThought(AmountExtractor)

    eval_res = []

    for idx, (att_, target_amount) in tqdm(enumerate(dataset_), total=len(dataset_)):
        pred = basic_predictor(file=att_)

        eval_res.append([att_, pred.amount, target_amount, pred.amount == target_amount])

        if pred.amount != target_amount:
            print(f"Misclassified id {idx}, true: {target_amount},  \t pred: {pred.amount}")

    return eval_res


@dataclass
class Config:
    max_refine_iters: int = 5
    temperature: float = 0.0
    parallel_workers: int = 8
    # Numeric equality tolerance in euros
    amount_tol: float = 0.002
    verbose: bool = True


##############################################################################
# 1.  DSPy SIGNATURES
##############################################################################


class AmountExtractor(dspy.Signature):
    """
    Given a file containing a Dutch invoice, extract the amount excluding taxes (btw).
    """

    file: Attachments = dspy.InputField()
    amount: float = dspy.OutputField(desc="Amount excluding taxes (ex-BTW) in euros")


class PromptReviewer(dspy.Signature):
    """
    You're given a prompt with instructions to extract the amount without taxes from the given
    Dutch invoice, the amount that was incorrectly extracted, the reasoning why that amount was
    picked and the correct amount.
    Your goal is to review the document, the reasoning and the incorrect amount step by step
    in order to rewrite the prompt so that the correct amount would have been returned.
    """

    prompt_instructions: str = dspy.InputField(desc="Original instructions")
    invoice: Attachments = dspy.InputField(desc="Dutch invoice")
    incorrect_amount: float = dspy.InputField(
        desc="Original incorrect extracted amount without taxes"
    )
    input_reasoning: str = dspy.InputField(
        desc="Original reasoning that led to the incorrect amount being extracted"
    )
    correct_amount: float = dspy.InputField(
        desc="Correct amount excluding taxes that should have been returned"
    )
    history: str = dspy.InputField(desc="JSON list of previous revised prompts")

    prompt: str = dspy.OutputField(
        desc="Improved prompt to extract the correct amount without taxes"
    )


class PromptMerger(dspy.Signature):
    """
    You're given a set of prompts (base_prompt and new_prompt) for extracting the amount
    without taxes from Dutch invoices.
    The base prompt is the one we want to extend with any added details or information from the new
    prompt.
    Review each of the prompts and merge them into one prompt that covers all the aspects from
    both prompts.
    """

    base_prompt: str = dspy.InputField(
        desc="The base prompt containing the most details and information"
    )
    new_prompt: str = dspy.InputField(
        desc="A new prompt that contains information for a particular example"
    )
    prompt: str = dspy.OutputField(desc="Merged prompt")


##############################################################################
# 2.  HELPER UTILITIES
##############################################################################


def numbers_are_close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, abs_tol=tol)


def parse_float(value: str) -> float:
    """
    Normalise Dutch/intl number formatting into float.
    Accepts '€ 1.234,56'  -> 1234.56
    """
    value = re.sub(r"[^\d,.-]", "", value)
    if value.count(",") == 1 and value.count(".") > 0:
        # 1.234,56 style: drop thousands sep then replace comma
        value = value.replace(".", "").replace(",", ".")
    elif value.count(",") == 1 and value.count(".") == 0:
        value = value.replace(",", ".")
    return float(value)


##############################################################################
# 3.  CORE MODULES
##############################################################################


class Evaluator(dspy.Module):
    def __init__(self, model, cfg: Config | None = None, callbacks=None) -> None:
        super().__init__(callbacks)
        self.model = model
        self.cfg = cfg
        self.extractor = dspy.ChainOfThought(AmountExtractor)
        # We will dynamically set extractor.prompt in forward

    def forward(self, prompt: str, file: Attachments) -> dspy.Prediction:
        self.extractor.predict.signature.instructions = prompt  # type: ignore

        # return {"amount": resp.amount, "reasoning": resp.reasoning}
        resp: dspy.Prediction = self.extractor(file=file)

        return resp


class Refiner(dspy.Module):
    def __init__(self, model, cfg: Config | None = None, callbacks=None) -> None:
        super().__init__(callbacks)
        self.model = model
        self.cfg = cfg
        self.reviewer = dspy.ChainOfThought(PromptReviewer)

    def forward(
        self,
        cur_prompt: str,
        file: Attachments,
        wrong_amount: float,
        rationale: str,
        correct_amount: float,
        history: list[str],
    ) -> dspy.Prediction:

        review: dspy.Prediction = self.reviewer(
            prompt_instructions=cur_prompt,
            invoice=file,
            incorrect_amount=wrong_amount,
            input_reasoning=rationale,
            correct_amount=correct_amount,
            history=json.dumps(history),
        )
        return review


class Merger(dspy.Module):
    def __init__(self, callbacks=None) -> None:
        super().__init__(callbacks)
        self.merger = dspy.ChainOfThought(PromptMerger)

    def forward(self, base: str, new: str) -> dspy.Prediction:
        merged: dspy.Prediction = self.merger(base_prompt=base, new_prompt=new)
        return merged


##############################################################################
# 4.  OPTIMISER ORCHESTRATOR
##############################################################################


class InvoicePromptOptimiser(dspy.Module):
    """
    Runs prompt-refinement over a labelled dataset of (invoice, gold_amount).
    """

    def __init__(self, model, base_prompt: str, cfg: Config | None = None, callbacks=None) -> None:
        super().__init__(callbacks)

        self.base_prompt = base_prompt
        self.cfg = cfg or Config()

        self.eval_mod = Evaluator(model, cfg)
        self.refine_mod = Refiner(model, cfg)
        self.merge_mod = Merger()

    # ------------------------------ PUBLIC API ------------------------------

    def optimise(self, dataset: list[tuple[Attachments, float]]) -> dict:
        prompt = self.base_prompt
        flagged: list[int] = []

        for idx, (invoice_file, gold_amount) in tqdm(enumerate(dataset)):
            if self.cfg.verbose:
                print(f"\n▶ Sample {idx}")

            history: list[str] = []
            for attempt in range(self.cfg.max_refine_iters + 1):
                eval_pred = self.eval_mod(prompt, invoice_file)
                eval_pred_amount, eval_pred_reasoning = eval_pred.amount, eval_pred.reasoning

                if numbers_are_close(eval_pred_amount, gold_amount, self.cfg.amount_tol):
                    if self.cfg.verbose:
                        print(f"  ✅ Correct on attempt {attempt}")

                    if attempt > 0:  # Only merge if we actually refined
                        print("\t\tMerging new prompt:")
                        print(f"\t\t\tOld: {history[-1]}")
                        print(f"\t\t\tCurrent: {prompt}")
                        merge_res = self.merge_mod(prompt, history[-1])
                        print(f"\t\t\tNew: {merge_res.prompt}")
                        prompt = merge_res.prompt
                    break

                if attempt == self.cfg.max_refine_iters:
                    if self.cfg.verbose:
                        print("  ❌ Failed after max iterations.")
                    flagged.append(idx)
                    break

                refine_pred = self.refine_mod(
                    prompt,
                    invoice_file,
                    eval_pred_amount,
                    eval_pred_reasoning,
                    gold_amount,
                    history,
                )
                history.append(refine_pred.prompt)
                prompt = refine_pred.prompt

        acc = self._evaluate(prompt, dataset)
        return dict(final_prompt=prompt, flagged=flagged, accuracy=acc)

    # ------------------------------ INTERNAL ------------------------------

    def _evaluate(self, prompt: str, dataset: list[tuple[Attachments, float]]) -> float:
        correct = 0

        def _worker(pair):
            file, gold = pair
            amt = self.eval_mod(prompt, file)["amount"]
            return numbers_are_close(amt, gold, self.cfg.amount_tol)

        with cf.ThreadPoolExecutor(self.cfg.parallel_workers) as ex:
            results = list(ex.map(_worker, dataset))
        correct = sum(results)
        return correct / len(dataset)


class DummyModel:
    """Always returns 0.0 to force the refiner to work."""

    def __call__(self, prompt: str, temperature: float = 0.0):
        return {"answer": "0.0", "reasoning": "Totale btw = totale bedrag."}


if __name__ == "__main__":
    data_path = get_root() / "data" / "invoices"

    # df = pd.read_parquet(data_path / "wrong_amounts.parquet")
    grouped_df = pd.read_parquet(data_path / "grouped_df.parquet")

    # files = set(data_path.rglob("*.pdf", case_sensitive=False))
    # for item in df["file_path"].str.replace("/data/dataset_scopes/scope 3", str(data_path)):
    #     print(Path(item) in files, item)

    # df["file_path"] = df["file_path"].str.replace("/data/dataset_scopes/scope 3", str(data_path))

    # dataset = make_dataset(df)
    # print(f"{len(dataset)=}")

    # initial_eval(dataset)
    dataset_tuple = make_dataset_tuple(grouped_df)

    base_prompt = (
        "Given a file containing a Dutch invoice, extract the amount excluding taxes (btw)."
    )

    # dataset_tuple = make_dataset_tuple(df)
    initial_res = initial_eval_tuples(dataset_tuple)

    optimizer = InvoicePromptOptimiser(DummyModel(), base_prompt)

    result = optimizer.optimise(dataset_tuple)

    print("\n=== FINAL PROMPT ===")
    print(result["final_prompt"])
    print("Flagged samples:", result["flagged"])
    print("Accuracy:", result["accuracy"])
