"""LongBench at 32K tokens in the Protocol-B setting of the BellowsKV paper, without the research harness.

Protocol B (the evaluation harness of xKV / ShadowKV) on 11 English LongBench tasks:
  prompts : LongBench's task templates. The QA and summarization templates go inside the harness's raw Llama-3 chat
            format (system "You are a helpful assistant", no newline after the headers). Tokenized with BOS. Only the
            prompts with 4096 < tokens <= 32768 are kept, in dataset order. There is no truncation.
  decoding: greedy, one prompt at a time, up to the task's gen_len tokens (the harness's effective lengths), stops at
            the model's EOS tokens. The text is decoded as the harness did it (`decode` below).
  metrics : LongBench's metrics (QA F1, ROUGE-L from the `rouge` 1.0.1 package, retrieval accuracy, code edit
            similarity), the max over the reference answers, averaged per task. Retention = 100 x method / dense for
            each task, then the mean over the 11 tasks.
Data: THUDM/LongBench `data.zip` at a pinned revision, read directly (no dataset script).
tests/check_flash_longbench.py checks that the prompt token ids and the scores are identical to the harness.
The templates and metrics are LongBench's (THUDM, MIT license), as the ShadowKV / xKV harness uses them (Apache-2.0).
"""
from __future__ import annotations

import hashlib
import json
import re
import string
import zipfile
from collections import Counter

REPO, REVISION = "THUDM/LongBench", "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "musique", "narrativeqa", "gov_report", "qmsum",
         "passage_retrieval_en", "lcc", "repobench-p", "2wikimqa"]
SHORT = dict(zip(TASKS, ["Qasper", "MFQA", "HotpotQA", "MuSiQue", "NrtvQA", "GovRep", "QMSum", "PsgRet", "LCC",
                         "RepoP", "2Wiki"]))
CHAT = [t for t in TASKS if t not in ("lcc", "repobench-p")]       # the code tasks use the bare template
DATALEN, MIN_LEN = 32768, 4096

TEMPLATE = {
    'qasper': 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    'multifieldqa_en': 'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    'hotpotqa': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    'musique': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    'narrativeqa': 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    'gov_report': 'You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:',
    'qmsum': 'You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:',
    'passage_retrieval_en': 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    'lcc': 'Please complete the code given below. \n{context}Next line of code:\n',
    'repobench-p': 'Please complete the code given below. \n{context}{input}Next line of code:\n',
    '2wikimqa': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
}
LLAMA3_CHAT = '<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
# The harness's effective generation lengths. Its `'qa' in name` rule comes before its LongBench table, so qasper,
# multifieldqa_en and narrativeqa run with 32 tokens (the LongBench table has 128 / 64 / 128).
GEN_LEN = {'qasper': 32, 'multifieldqa_en': 32, 'hotpotqa': 32, 'musique': 32, 'narrativeqa': 32, 'gov_report': 512,
           'qmsum': 512, 'passage_retrieval_en': 32, 'lcc': 64, 'repobench-p': 64, '2wikimqa': 32}

# Llama-3.1-8B-Instruct tokenizer: number of prompts kept and the sha256 (first 16 hex) of their token ids.
# This is the paper's prompt set. build() warns when a tokenizer gives a different set.
FINGERPRINT = {
    'qasper': (122, '719c689805c0616a'), 'multifieldqa_en': (110, '8bcb3f99895c3520'),
    'hotpotqa': (195, '7487e28bda09c14c'), 'musique': (200, '7efdede2835d8e15'),
    'narrativeqa': (115, '2fcf02f8a5553e61'), 'gov_report': (177, '422a96bb3c99d7e5'),
    'qmsum': (199, '8deccd84e25a4c32'), 'passage_retrieval_en': (200, '8434c72aa9639a30'),
    'lcc': (93, '6e49be6519b12cdb'), 'repobench-p': (442, '64b5fb02a95a86a7'), '2wikimqa': (175, '3859bed4484a6de8'),
}


# ----------------------------------------------------------------------------------------------- data
def data_zip(path: str | None = None) -> str:
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, "data.zip", repo_type="dataset", revision=REVISION)


def records(task: str, zip_path: str | None = None) -> list[dict]:
    """The task's test records in file order (the order of the LongBench dataset script)."""
    with zipfile.ZipFile(data_zip(zip_path)) as z, z.open(f"data/{task}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(task: str, rec: dict) -> str:
    t = LLAMA3_CHAT.format(ctx=TEMPLATE[task]) if task in CHAT else TEMPLATE[task]
    return t.format(**rec)


def fingerprint(ids) -> str:
    h = hashlib.sha256()
    for x in ids:
        h.update(x.numpy().astype("<i4").tobytes()); h.update(b"|")
    return h.hexdigest()[:16]


def build(tok, tasks=TASKS, datalen: int = DATALEN, min_len: int = MIN_LEN, zip_path: str | None = None, log=print):
    """Tokenized prompts per task: {task: dict(ids=[int32 tensors], answers, classes, index, gen_len)}.
    index = the record's position in the task file."""
    import torch
    out = {}
    for t in tasks:
        recs = records(t, zip_path)
        enc = tok([prompt_text(t, r) for r in recs]).input_ids            # BOS added, as tok(text) does
        keep = [k for k, x in enumerate(enc) if min_len < len(x) <= datalen]
        ids = [torch.tensor(enc[k], dtype=torch.int32) for k in keep]
        out[t] = dict(ids=ids, answers=[recs[k]["answers"] for k in keep], classes=[recs[k]["all_classes"] for k in keep],
                      index=keep, gen_len=GEN_LEN[t])
        fp, ref = fingerprint(ids), FINGERPRINT.get(t)
        same = ref is not None and (len(ids), fp) == tuple(ref) and (datalen, min_len) == (DATALEN, MIN_LEN)
        log(f"[longbench] {t:22s} {len(ids):4d} prompts (of {len(recs)}), gen_len {GEN_LEN[t]:3d}, "
            f"{'= the paper prompt set' if same else 'fingerprint ' + fp + ' (not the paper prompt set)'}")
    return out


def clean_up_tokenization(s: str) -> str:
    """transformers 4.x `clean_up_tokenization`. The harness ran transformers 4.57, which applies it because the
    Llama-3.1 tokenizer config sets clean_up_tokenization_spaces = True. transformers 5 skips it for BPE tokenizers."""
    return (s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
            .replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re"))


def decode(tok, toks) -> str:
    """The prediction text as the harness got it (any transformers version): skip the special tokens, then the
    4.x cleanup. Code edit similarity compares raw strings, so the cleanup changes some lcc / repobench-p scores."""
    return clean_up_tokenization(tok.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=False))


# ----------------------------------------------------------------------------------------------- metrics
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower case, remove punctuation, articles and extra whitespace (the order of the LongBench code)."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def qa_f1(pred: str, gt: str, **_) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0
    prec, rec = 1.0 * same / len(p), 1.0 * same / len(g)
    return (2 * prec * rec) / (prec + rec)


def rouge_l(pred: str, gt: str, **_) -> float:
    try:
        from rouge import Rouge
    except ImportError as e:
        raise ImportError("gov_report / qmsum are scored with the `rouge` 1.0.1 package: pip install 'bellowskv[eval]'") from e
    try:
        return Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:          # as in the harness: 0 when rouge fails (empty hypothesis, recursion limit)
        return 0.0


def retrieval(pred: str, gt: str, **_) -> float:
    gid = re.findall(r"Paragraph (\d+)", gt)[0]
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else float(sum(1 for n in nums if n == gid) / len(nums))


def _fuzz_ratio(a: str, b: str) -> int:
    """fuzzywuzzy 0.18 `fuzz.ratio` with python-Levenshtein (the harness env): round(100 x normalized Indel similarity)."""
    if a == b:
        return 100
    if not a or not b:
        return 0
    try:
        from rapidfuzz.distance import Indel
    except ImportError as e:
        raise ImportError("lcc / repobench-p are scored with `rapidfuzz`: pip install 'bellowskv[eval]'") from e
    return int(round(100 * Indel.normalized_similarity(a, b)))


def code_sim(pred: str, gt: str, **_) -> float:
    line = next((x for x in pred.lstrip("\n").split("\n") if "`" not in x and "#" not in x and "//" not in x), "")
    return _fuzz_ratio(line, gt) / 100


METRIC = {"qasper": qa_f1, "multifieldqa_en": qa_f1, "hotpotqa": qa_f1, "musique": qa_f1, "narrativeqa": qa_f1,
          "2wikimqa": qa_f1, "gov_report": rouge_l, "qmsum": rouge_l, "passage_retrieval_en": retrieval,
          "lcc": code_sim, "repobench-p": code_sim}


def score_task(task: str, preds: list[str], answers: list[list[str]], classes=None) -> tuple[float, list[float]]:
    """Per-sample score = max over the reference answers; returns (mean, per-sample scores)."""
    fn = METRIC[task]
    classes = classes or [None] * len(preds)
    per = [max(fn(p, g, all_classes=c) for g in a) for p, a, c in zip(preds, answers, classes)]
    return sum(per) / len(per), per
