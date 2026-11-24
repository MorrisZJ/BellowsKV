"""bellowskv.flash.longbench metrics on hand-computed cases (CPU; needs the `eval` extra: rouge, rapidfuzz).
The full check against the research harness is tests/check_flash_longbench.py.
usage: python tests/test_longbench_metrics.py"""
from bellowskv.flash import longbench as LB


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_qa_f1():
    # "the cat sat." -> "cat sat" vs "cat": 1 common token, precision 1/2, recall 1
    assert close(LB.qa_f1("The cat sat.", "a cat"), 2 * 0.5 * 1.0 / 1.5)
    assert LB.qa_f1("dog", "cat") == 0
    assert close(LB.qa_f1("Paris, France", "paris france"), 1.0)


def test_retrieval():
    assert LB.retrieval("Paragraph 3", "Paragraph 3") == 1.0
    assert LB.retrieval("Paragraph 3 or Paragraph 12", "Paragraph 12") == 0.5
    assert LB.retrieval("no number", "Paragraph 4") == 0.0


def test_code_sim():
    # the first line without ` # // is compared: "x = foo(a, b)"
    assert LB.code_sim("```python\nx = foo(a, b)\n", "x = foo(a, b)") == 1.0
    # "y = 1" vs "y = 2": Indel distance 2 over 10 characters -> 80
    assert LB.code_sim("# comment\ny = 1", "y = 2") == 0.8
    assert LB.code_sim("// only comments", "z = 3") == 0.0


def test_rouge_l():
    assert close(LB.rouge_l("a b c", "a b c"), 1.0, 1e-6)
    assert LB.rouge_l("", "a b c") == 0.0          # empty hypothesis: rouge raises, scored 0 as in the harness


def test_score_task_and_decode_cleanup():
    avg, per = LB.score_task("hotpotqa", ["cat", "dog"], [["cat", "kitten"], ["cat"]])
    assert per == [1.0, 0] and avg == 0.5
    assert LB.clean_up_tokenization("a , b . c 's") == "a, b. c's"
    assert LB.GEN_LEN["qasper"] == 32 and LB.GEN_LEN["gov_report"] == 512


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("ALL_OK")
