#!/usr/bin/env python3
"""单章结构检查：字数、段落、句长、对话占比、开篇抓手、章末钩子。

用法：
    python3 novel/tools/chapter_check.py novel/03_正文/第001章_xxx.md
    python3 novel/tools/chapter_check.py --outline novel/02_大纲/细纲/第001章.md   # 检查细纲字段

参数取自 novel/config.json（缺省用内置值）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "chapter_cjk_min": 2200,
    "chapter_cjk_max": 3200,
    "paragraph_cjk_max": 150,
    "sentence_cjk_max": 40,
    "long_sentence_ratio_max": 0.15,
    "dialogue_ratio_min": 0.20,
    "opening_window": 300,
    "outline_required": ["本章目标", "阻碍", "本章变化", "爽点", "章末钩子", "伏笔", "场景/分镜", "出场人物", "情绪强度"],
}
WEAK_ENDINGS = ["新的一天", "明天", "未来", "一切才刚刚开始", "继续", "沉沉睡去", "夜色", "月光", "静静地"]
OPENING_STATIC = re.compile(r"^[^。！？]{0,30}(?:站在|坐在|躺在|望着|看着|想着|回忆)")
CJK = re.compile(r"[一-鿿]")
QUOTE = re.compile(r"“[^”]*”|「[^」]*」")


def cfg():
    p = ROOT / "config.json"
    c = dict(DEFAULTS)
    if p.exists():
        c.update(json.loads(p.read_text(encoding="utf-8")).get("chapter", {}))
    return c


def n(s):
    return len(CJK.findall(s))


def check_chapter(text, c):
    issues = []
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")).strip()
    paras = [p.strip() for p in body.split("\n") if p.strip()]
    total = n(body)
    if not c["chapter_cjk_min"] <= total <= c["chapter_cjk_max"]:
        issues.append(("blocking", "length", f"字数 {total}，目标 {c['chapter_cjk_min']}–{c['chapter_cjk_max']}"))
    for i, p in enumerate(paras, 1):
        if n(p) > c["paragraph_cjk_max"]:
            issues.append(("advisory", "long-paragraph", f"第 {i} 段 {n(p)} 字，手机阅读请拆段"))
    sents = [s for s in re.split(r"[。！？!?…]+", QUOTE.sub("", body)) if n(s) > 0]
    longs = [s for s in sents if n(s) > c["sentence_cjk_max"]]
    if sents and len(longs) / len(sents) > c["long_sentence_ratio_max"]:
        issues.append(("advisory", "long-sentences", f"长句占比 {len(longs)/len(sents):.0%}，例：{longs[0][:30]}…"))
    dlg = sum(n(m) for m in QUOTE.findall(body))
    ratio = dlg / total if total else 0
    if ratio < c["dialogue_ratio_min"]:
        issues.append(("advisory", "low-dialogue", f"对话占比 {ratio:.0%}，短剧化改编需要更多对话驱动"))
    if paras and OPENING_STATIC.search(paras[0]):
        issues.append(("blocking", "static-opening", "开篇是静态描写（站着、看着、想着），前 300 字要有事件或冲突"))
    if paras:
        last = paras[-1]
        for w in WEAK_ENDINGS:
            if w in last:
                issues.append(("advisory", "weak-ending", f"章末含“{w}”，检查是不是抒情收尾而不是钩子"))
                break
    stats = {"cjk": total, "paragraphs": len(paras), "dialogue_ratio": round(ratio, 2)}
    return issues, stats


def check_outline(text, c):
    issues = []
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # 忽略填写说明注释
    for field in c["outline_required"]:
        m = re.search(rf"^[ \t]*[-*]?[ \t]*\**{re.escape(field)}\**[ \t]*[:：][ \t]*(.*)$", text, re.M)
        if not m or not m.group(1).strip() or m.group(1).strip() in {"TODO", "待定", "-"}:
            issues.append(("blocking", "outline-field", f"细纲缺少字段：{field}"))
    return issues, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outline", action="store_true", help="按细纲模板检查字段")
    args = ap.parse_args()
    c = cfg()
    bad = 0
    for f in args.files:
        text = Path(f).read_text(encoding="utf-8")
        issues, stats = (check_outline if args.outline else check_chapter)(text, c)
        for lvl, rid, msg in issues:
            print(f"{f}\t[{lvl}] {rid}\t{msg}")
            bad += lvl == "blocking"
        status = "FAIL" if any(i[0] == "blocking" for i in issues) else "ok"
        print(f"{f}: {status} {stats}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
