#!/usr/bin/env python3
"""去 AI 味检查（写作 lint）。

用法：
    python3 novel/tools/deslop.py novel/03_正文/第001章_xxx.md [更多文件...]
    python3 novel/tools/deslop.py --json FILE

规则分两档：
- blocking：确定性的句式 / 元话语问题，出现即要求修改（退出码 1）。
- advisory：口水词 / 套话密度提示，按读感酌情修改。

规则清单的思路借鉴自 zenstory-ai/oh-story-claudecode（MIT）的 story-deslop、
LAY-lgtm/novel-writing-framework（MIT）的 AI 指纹指标、NousResearch/autonovel 的 ANTI-SLOP。
正则为本项目独立编写。
"""
import argparse
import json
import re
import sys
from pathlib import Path

# (id, 级别, 正则, 说明/改写方向, 每章容许次数)
RULES = [
    ("not-x-but-y", "blocking",
     r"(?<![要会能])不是[^。！？\n“”]{1,24}?[，,]\s*(?:而)?是",
     "“不是X，而是Y”：直接写Y，或用动作表现差异", 2),
    ("negation-parade", "blocking",
     r"没有[^，。！？\n]{1,8}[，,]\s*没有[^，。！？\n]{1,8}[，,。]",
     "排比式否定：改成一个具体动作", 0),
    ("voice-contrast", "blocking",
     r"(?:声音|语气|话)不(?:大|高|重|响)[，,]\s*却",
     "“声音不大，却……”：删掉评判，写听者的反应", 0),
    ("meta-talk", "blocking",
     r"第[一二三四五六七八九十百千零\d]+[章卷]|本章|上一章|下一章|前文|后文|卷[一二三四五六七八九十]",
     "正文出现元话语，读者会出戏：用具体事件指代", 0),
    ("summary-moral", "blocking",
     r"(?:这|那)(?:就是|便是|才是)[^。！？\n]{0,12}(?:意义|道理|代价|成长|答案|真相)",
     "替读者总结升华：删掉，让事件自己说话", 0),
    ("narrator-comment", "blocking",
     r"^不叫[^，。“”]{1,8}[，,]\s*叫[^，。“”]{1,8}。$|算是(?:默认|答应|回答|表态|认可|同意|回应)|这话不假|此话不假|言下之意|意思很明显|意思很清楚|显而易见",
     "叙述者跳出来点评或解释（替读者解读称呼、动作、沉默的含义）：删掉，让读者自己看出来", 0),
    ("realize-tell", "advisory",
     r"(?:他|她|我)(?:终于)?(?:明白|意识到|懂得)了",
     "顿悟式陈述：改成行为变化", 2),
]

# 口水词：按每千字密度提示
CLICHES = [
    "缓缓", "微微", "不禁", "深吸一口气", "眼底闪过", "眼中闪过", "嘴角勾起", "嘴角上扬",
    "一丝不易察觉", "心中升起", "心中涌起", "意味深长", "空气中弥漫", "命运的齿轮",
    "不由自主", "下意识", "五味杂陈", "内心深处", "莫名的", "淡淡的", "仿佛", "宛如",
    "这一刻", "某种", "说不出的", "难以言喻", "不知不觉", "与此同时", "氤氲", "弥漫着",
]
CLICHE_PER_KCHAR = 3.0     # 每千字超过此数提示
DASH_LIMIT = 12            # 叙事中破折号上限（对话内不计）

CJK = re.compile(r"[一-鿿]")
QUOTE = re.compile(r"“[^”]*”|「[^」]*」")


def strip_md(text: str) -> str:
    # 去掉 Markdown 标题行，避免章节标题误报元话语
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def check(text: str):
    body = strip_md(text)
    lines = body.splitlines()
    findings = []
    for rid, level, pat, tip, allow in RULES:
        rx = re.compile(pat)
        hits = []
        for ln, line in enumerate(lines, 1):
            # 叙述者点评只查叙述部分，人物台词里说"算是默认"是允许的
            target = QUOTE.sub("", line) if rid == "narrator-comment" else line
            for m in rx.finditer(target):
                hits.append((ln, m.start() + 1, m.group(0)))
        if len(hits) > allow:
            for ln, col, frag in hits:
                findings.append({"line": ln, "col": col, "level": level, "rule": rid,
                                 "hit": frag, "tip": f"{tip}（本章 {len(hits)} 次，容许 {allow}）"})

    narration = QUOTE.sub("", body)
    dashes = narration.count("——")
    if dashes > DASH_LIMIT:
        findings.append({"line": 0, "col": 0, "level": "blocking", "rule": "dash-overuse",
                         "hit": f"叙事破折号 {dashes} 处",
                         "tip": f"上限 {DASH_LIMIT}，改成逗号、句号或拆句"})

    nchar = len(CJK.findall(body)) or 1
    total = 0
    detail = {}
    for w in CLICHES:
        c = body.count(w)
        if c:
            detail[w] = c
            total += c
    density = total * 1000 / nchar
    if density > CLICHE_PER_KCHAR:
        top = "、".join(f"{k}×{v}" for k, v in sorted(detail.items(), key=lambda x: -x[1])[:8])
        findings.append({"line": 0, "col": 0, "level": "advisory", "rule": "cliche-density",
                         "hit": top, "tip": f"口水词密度 {density:.1f}/千字（阈值 {CLICHE_PER_KCHAR}）"})
    return findings, {"cjk": nchar, "dash": dashes, "cliche_density": round(density, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    blocking = 0
    report = {}
    for f in args.files:
        findings, stats = check(Path(f).read_text(encoding="utf-8"))
        report[f] = {"stats": stats, "findings": findings}
        blocking += sum(1 for x in findings if x["level"] == "blocking")
        if not args.json:
            for x in findings:
                print(f"{f}:{x['line']}:{x['col']}\t[{x['level']}] {x['rule']}\t({x['hit']})\t→ {x['tip']}")
            print(f"{f}: 字数 {stats['cjk']} · 破折号 {stats['dash']} · 口水词密度 {stats['cliche_density']}/千字 · "
                  f"{sum(1 for x in findings if x['level']=='blocking')} blocking")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
