"""Run a pretrained OpenTSLM model on HAR-CoT under every probe condition.

Writes one JSON line per (condition, sample) with the raw rationale and the
parsed label, and resumes from an existing output file: a Colab disconnect
only costs the batch that was running.

Needs a GPU, the opentslm package and access to the gated base LLM on Hugging
Face (see the README).

    python scripts/run_probe.py --model OpenTSLM/llama-3.2-1b-har-sp
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEED  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import extract_label  # noqa: E402
from src.perturb import CONDITIONS, BY_NAME, build_inputs, condition_rng  # noqa: E402
from src.sampling import assign_donors, stratified_subset  # noqa: E402


def load_model(repo_id):
    from opentslm.model.llm.OpenTSLM import OpenTSLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Same rule as OpenTSLM's HAR demo: SoftPrompt checkpoints carry LoRA weights.
    model = OpenTSLM.load_pretrained(repo_id, enable_lora="-sp" in repo_id, device=device)
    model.eval()
    return model


def prompt_frame():
    """Pre- and post-prompt exactly as OpenTSLM's dataset builds them."""
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

    # Both methods ignore their arguments; calling them unbound avoids loading
    # the 465 MB training split that instantiating the dataset would trigger.
    pre = HARCoTQADataset._get_pre_prompt(None, None).strip()
    post = HARCoTQADataset._get_post_prompt(None, None).strip()
    return pre, post


def to_model_sample(series, texts, pre_prompt, post_prompt):
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    sample = {
        "pre_prompt": pre_prompt,
        "time_series_text": texts,
        "time_series": [axis for axis in series],
        "post_prompt": post_prompt,
    }
    return extend_time_series_to_match_patch_size_and_aggregate([sample], patch_size=PATCH_SIZE)[0]


@torch.inference_mode()
def generate(model, samples, max_new_tokens):
    """Greedy generation for a list of samples.

    OpenTSLM's own batched `generate` right-pads sequences of different length,
    which shifts where a decoder-only model starts generating for every sample
    but the longest. Prompts here differ in length (the stated statistics
    tokenise differently), so for SoftPrompt models the probe embeds each sample
    on its own and left-pads the batch itself. Flamingo models fall back to one
    sample per call.
    """
    if type(model).__name__ != "OpenTSLMSP" or len(samples) == 1:
        return [model.generate([s], max_new_tokens=max_new_tokens, do_sample=False)[0] for s in samples]

    embeds, masks = zip(*(model.pad_and_apply_batch([s]) for s in samples))
    longest = max(e.size(1) for e in embeds)
    embeds = torch.cat([
        torch.cat([e.new_zeros(1, longest - e.size(1), e.size(2)), e], dim=1) for e in embeds
    ])
    masks = torch.cat([
        torch.cat([m.new_zeros(1, longest - m.size(1)), m], dim=1) for m in masks
    ])
    tokenizer = model.tokenizer
    out = model.llm.generate(
        inputs_embeds=embeds,
        attention_mask=masks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def check_batching(model, samples, max_new_tokens):
    """Batched and one-at-a-time generation must agree before the batched path is trusted."""
    batched = generate(model, samples, max_new_tokens)
    single = [generate(model, [s], max_new_tokens)[0] for s in samples]
    same_label = sum(extract_label(a) == extract_label(b) for a, b in zip(batched, single))
    same_text = sum(a.strip() == b.strip() for a, b in zip(batched, single))
    print(f"Batching check on {len(samples)} samples: "
          f"{same_text} identical rationales, {same_label} identical labels")
    return same_label == len(samples)


def already_done(path):
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["condition"], r["index"]))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="OpenTSLM/llama-3.2-1b-har-sp")
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument("--per-class", default=50, type=int)
    parser.add_argument("--conditions", nargs="+", default=[c.name for c in CONDITIONS],
                        choices=[c.name for c in CONDITIONS])
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=400, type=int)
    parser.add_argument("--limit", default=None, type=int,
                        help="only the first N subset samples (smoke test)")
    parser.add_argument("--check-batching", action="store_true",
                        help="compare batched and single generation on one batch, then exit")
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    out = args.out or ROOT / "results" / f"predictions_{args.model.split('/')[-1]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    signals, labels = load_split(args.data_dir, "test")
    subset = stratified_subset(labels, args.per_class, SEED)
    donors = assign_donors(labels, subset, SEED + 1)
    if args.limit:
        subset, donors = subset[: args.limit], donors[: args.limit]
    donor_of = dict(zip(subset.tolist(), donors.tolist()))

    manifest = {
        "model": args.model, "per_class": args.per_class, "seed": SEED,
        "max_new_tokens": args.max_new_tokens, "subset": subset.tolist(), "donors": donors.tolist(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))

    model = load_model(args.model)
    pre_prompt, post_prompt = prompt_frame()

    if args.check_batching:
        original = BY_NAME["original"]
        samples = [
            to_model_sample(*build_inputs(signals[i], None, original, None), pre_prompt, post_prompt)
            for i in subset[: args.batch_size].tolist()
        ]
        ok = check_batching(model, samples, args.max_new_tokens)
        sys.exit(0 if ok else "Batched generation disagrees: rerun with --batch-size 1")

    done = already_done(out)

    for c_idx, condition in enumerate(CONDITIONS):
        if condition.name not in args.conditions:
            continue
        todo = [i for i in subset.tolist() if (condition.name, i) not in done]
        print(f"\n== {condition.name}: {len(todo)} to run ({condition.question})")
        start = time.time()
        for b in range(0, len(todo), args.batch_size):
            batch_idx = todo[b: b + args.batch_size]
            samples = []
            for i in batch_idx:
                donor = signals[donor_of[i]]
                series, texts = build_inputs(signals[i], donor, BY_NAME[condition.name],
                                             condition_rng(c_idx, i, SEED))
                samples.append(to_model_sample(series, texts, pre_prompt, post_prompt))
            outputs = generate(model, samples, args.max_new_tokens)
            with out.open("a", encoding="utf-8") as f:
                for i, text in zip(batch_idx, outputs):
                    f.write(json.dumps({
                        "condition": condition.name,
                        "index": i,
                        "label": str(labels[i]),
                        "donor_index": donor_of[i],
                        "donor_label": str(labels[donor_of[i]]),
                        "prediction": extract_label(text),
                        "output": text,
                    }) + "\n")
            done_now = b + len(batch_idx)
            rate = (time.time() - start) / done_now
            print(f"  {done_now}/{len(todo)}  ({rate:.1f} s/sample)", flush=True)

    print(f"\nPredictions in {out}")


if __name__ == "__main__":
    main()
