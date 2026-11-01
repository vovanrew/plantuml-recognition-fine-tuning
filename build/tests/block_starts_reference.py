"""PlantUML 1.2025.9's rule for where a diagram block starts, written again from its source (tag
v1.2025.9), as the reference that build/tests/test_pool.py checks build/pool.py's block rule against.

  utils/StartUtils.java        beforeStartUml, startsWithSymbolAnd, patternFilename
  core/DiagramType.java        getTypeFromArobaseStart (the block keyword may be written with @
                               or with a backslash, in any letter case, after any Java whitespace)
  preproc/UncommentReadLine.java  a block start may follow a prefix of non-word characters
                               (" * @startuml", "// @startuml"); the prefix is then cut from the
                               lines that follow
  BlockUmlBuilder.java         a block runs from a start line to the next end line
  BlockUml.java                getFileOrDirname: the name after the keyword names the PNG
!include and the preprocessor are not followed. Moved here from the render-pipeline item's working
scripts, its functions unchanged.
"""
import re

JAVA_WS = set(" \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f              　")
WORD = re.compile(r"[A-Za-z0-9_~]")
TYPES = ("bpm", "board", "creole", "chronology", "chen", "dot", "ditaa", "def", "ebnf", "flow", "files", "gantt",
         "git", "hcl", "jcckit", "json", "latex", "math", "mindmap", "nwdiag", "project", "regex", "salt", "uml",
         "wire", "wbs", "yaml")
SP = r"\s 　"
Q = "\"“”«»"
FILENAME = re.compile(rf"^[@\\]start[^{SP}{{}}{Q}]+[{SP}{{][{SP}{Q}]*([^{Q}]*?)[{SP}}}{Q}]*$", re.I)


def symbol_and(keyword, text):
    for i, c in enumerate(text):
        if c in JAVA_WS:
            continue
        if c not in "@\\":
            return False
        return text[i + 1:i + 1 + len(keyword)] == keyword
    return False


def before_start(s):
    inside = False
    for i, ch in enumerate(s):
        if symbol_and("start", s[i:]):
            return s[:i]
        if inside:
            if ch == ">":
                inside = False
            continue
        if ch == "<":
            inside = True
        elif WORD.match(ch):
            return None
    return None


def is_start(s):
    for i, c in enumerate(s):
        if c in JAVA_WS:
            continue
        if c not in "@\\":
            return False
        rest = s[i + 1:].lower()
        return rest.startswith("start") and rest[5:].startswith(TYPES)
    return False


def name_of(first_line):
    m = FILENAME.search(first_line.strip("".join(JAVA_WS) + " "))
    if not m:
        return None
    name = m.group(1).split(",")[0]
    if any(c in name for c in "<>|"):
        return None
    if name.startswith("file://"):
        name = name[7:]
    name = re.sub(r"\.\w\w\w$", "", name)
    return name or None


def blocks(text):
    """First lines of the blocks PlantUML would build, in order."""
    header, current, found = None, None, []
    for raw in re.split(r"\r\n|\r|\n", text):
        tmp = before_start(raw)
        if tmp is not None:
            header = tmp
        line = raw
        if header is not None and header.startswith(raw):
            line = ""
        elif header is not None and raw.startswith(header):
            line = raw[len(header):]
        if is_start(line):
            current = line
        if symbol_and("end", line) and current is not None:
            found.append(current)
            current = None
    return found
