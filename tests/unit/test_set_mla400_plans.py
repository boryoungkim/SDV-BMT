"""tools/mobilint/set_mla400_plans.py: the committed MLA400 plans are its output for the 5,000-image input, and its
comment rewrite finds the image count even when a comment line break splits it."""

from __future__ import annotations

import subprocess
import sys

from tests._helpers import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools" / "mobilint"))
import set_mla400_plans as S  # noqa: E402


def test_committed_plans_match_the_tool_for_the_val2017_input():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "mobilint" / "set_mla400_plans.py"),
                           "--input", "coco-val2017", "--check"], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout == "", proc.stdout


def test_the_image_count_is_rewritten_across_line_breaks_and_keeps_them():
    first = ("# head tail on the host) on the first\n# 1000 COCO val2017 images at 416, nominal\n"
             "# ... on the\n# first 1000 COCO val2017 images, closed loop\n")
    full = S.rewrite(first, "coco-val2017", S.COUNT_WORDS["coco-val2017"], None)
    assert full == ("# head tail on the host) on the full\n# 5,000 COCO val2017 images at 416, nominal\n"
                    "# ... on the\n# full 5,000 COCO val2017 images, closed loop\n")
    assert S.rewrite(full, "coco-val2017-first1000", S.COUNT_WORDS["coco-val2017-first1000"], None) == first
    first_version = "# on the all 5,000\n# COCO val2017 images, nominal\n"  # the wording of the tool's first version
    assert S.rewrite(first_version, "coco-val2017", S.COUNT_WORDS["coco-val2017"], None) == \
        "# on the full 5,000\n# COCO val2017 images, nominal\n"
