#!/usr/bin/env python3
"""声线一致性检查：把每个人物在所有章节里的台词抽出来，放在一起对照，并按声线规则查违规。

用法：
    python3 novel/tools/voice_check.py                    # 检查 03_正文 下全部章节
    python3 novel/tools/voice_check.py --who 老陈          # 只看某个人的全部台词
    python3 novel/tools/voice_check.py --dump             # 输出每人全部台词（审稿用）

说话人识别规则（启发式）：同一段落里，引号外的叙述部分最先出现的人物称呼。
识别不了的台词归入"?"，审稿时人工看。规则见 01_设定/人物/声线规则.json。
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUOTE = re.compile(r"“([^”]*)”")
CJK = re.compile(r"[一-鿿]")


def load_rules():
    return json.loads((ROOT / "01_设定/人物/声线规则.json").read_text(encoding="utf-8"))["characters"]


NON_SPEECH = re.compile(r"写着|喷着|印着|名字|叫作|叫做|外号|显示|弹出|改成|标题|招牌|群里")
PRONOUN_SPEECH = re.compile(r"[他她](?:说|问|喊|道|开口|回头|压着|小声|突然)")


def find_alias(text, rules, last=False):
    best, pos = None, (-1 if last else 10 ** 9)
    for name, r in rules.items():
        for a in r["aliases"]:
            i = text.rfind(a) if last else text.find(a)
            if i != -1 and ((last and i > pos) or (not last and i < pos)):
                best, pos = name, i
    return best


def extract(rules):
    """说话人推断：①引号外出现人名 → 该人；②“他说/她问”等 → 最近提到的人；
    ③纯对白段 → 按一问一答交替，取倒数第二个说话人；否则记为“?”。"""
    lines = defaultdict(list)
    for f in sorted((ROOT / "03_正文").glob("*.md")):
        ch = f.name.split("_")[0]
        last_named, history = None, []
        for para in f.read_text(encoding="utf-8").split("\n"):
            para = para.strip()
            if not para or para.startswith("#") or para == "---":
                if para == "---":
                    history = []
                continue
            qs = QUOTE.findall(para)
            outside = QUOTE.sub("", para)
            if not qs:
                last_named = find_alias(para, rules, last=True) or last_named
                continue
            if NON_SPEECH.search(outside) and not PRONOUN_SPEECH.search(outside):
                last_named = find_alias(outside, rules, last=True) or last_named
                continue
            who = find_alias(outside, rules)
            if not who and PRONOUN_SPEECH.search(outside):
                who = last_named
            if not who and not outside.strip(" ，。：") and len(history) >= 2 and history[-1] != history[-2]:
                who = history[-2]
            who = who or "?"
            for q in qs:
                lines[who].append((ch, q))
            if who != "?":
                history.append(who)
                last_named = find_alias(outside, rules, last=True) or who
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who")
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    rules = load_rules()
    lines = extract(rules)
    bad = 0
    for who in sorted(lines, key=lambda w: (w == "?", w)):
        if a.who and who != a.who:
            continue
        qs = lines[who]
        r = rules.get(who)
        print(f"\n== {who}：{len(qs)} 句台词")
        if r:
            for ch, q in qs:
                for s in re.split(r"[。！？!?…]+", q):
                    if len(CJK.findall(s)) > r["max_sentence"]:
                        print(f"  [超长] {ch} 「{s}」 {len(CJK.findall(s))}字 > {r['max_sentence']}")
                        bad += 1
                for w in r["banned"]:
                    if w in q:
                        print(f"  [禁词] {ch} 「{q}」 含“{w}”")
                        bad += 1
            hits = {c: sum(q.count(c) for _, q in qs) for c in r["catchphrases"]}
            print("  口头禅：" + "，".join(f"{k}×{v}" for k, v in hits.items()))
        if a.dump or a.who:
            for ch, q in qs:
                print(f"  {ch}  {q}")
    print(f"\n声线违规 {bad} 处（“?” 为未识别说话人，请人工核对）")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
