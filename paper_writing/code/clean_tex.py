#!/usr/bin/env python3
"""Strip comments from the Overleaf paper sources (the repo must contain clean LaTeX only).

Rules per line: find the first unescaped '%'. A line that is only a comment is dropped
(blank-line structure is preserved: at most one empty line in a row). If code precedes the
'%' with whitespace in between, the comment and the whitespace are removed. If the '%'
directly follows code (e.g. `{%`, used to suppress the line-end space), a bare '%' is kept.
For .bib files, only full-line comments are removed.
Usage: python clean_tex.py [repo]"""
import glob
import os
import re
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"
UNESC = re.compile(r"(?<!\\)(?:\\\\)*%")


def clean_line(line):
    m = UNESC.search(line)
    if not m:
        return line
    pct = m.end() - 1
    code = line[:pct]
    if not code.strip():
        return None
    if code != code.rstrip():
        return code.rstrip()
    return code + "%"


def clean(path, bib=False):
    out, prev_blank = [], False
    for line in open(path, encoding="utf-8").read().split("\n"):
        if bib:
            new = None if line.lstrip().startswith("%") else line
        else:
            new = clean_line(line)
        if new is None:
            continue
        blank = not new.strip()
        if blank and prev_blank:
            continue
        out.append(new)
        prev_blank = blank
    text = "\n".join(out).strip("\n") + "\n"
    open(path, "w", encoding="utf-8").write(text)


for p in [os.path.join(repo, "main.tex")] + glob.glob(os.path.join(repo, "latex", "**", "*.tex"), recursive=True):
    clean(p)
for p in glob.glob(os.path.join(repo, "*.bib")):
    clean(p, bib=True)
print("cleaned", repo)
