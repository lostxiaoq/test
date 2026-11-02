#!/usr/bin/env python3
"""事实账本：校验并提交每章的状态增量，再重新生成「上下文卡」。

思路借鉴：InkOS 的 Settler（增量 delta + 宿主校验）、webnovel-writer 的 CHAPTER_COMMIT、
oh-story 的「作者真相 / 读者已知」分离与派生视图。代码为本项目独立编写。

用法：
    python3 novel/tools/state_commit.py init                       # 初始化空账本
    python3 novel/tools/state_commit.py check  delta/第001章.json   # 只校验不写入
    python3 novel/tools/state_commit.py commit delta/第001章.json   # 校验后写入并重建上下文卡
    python3 novel/tools/state_commit.py render                     # 只重建 追踪/上下文.md

state/ 下的 JSON 是唯一权威源；追踪/上下文.md 由脚本生成，不要手改。

delta 格式（每章写完由「记账」步骤产出）：
{
  "chapter": 1,
  "title": "章名",
  "story_date": "2019-01-03",
  "summary": "一两句话本章发生了什么",
  "change_type": "局面|权力|关系|认知",
  "payoff_type": "打脸|碾压|逆转|升级|揭秘|救场|收获|无",
  "hook_type": "危机|悬念|反转|挑衅|倒计时|抉择",
  "events": ["..."],
  "characters": [
    {"id": "C001", "name": "林舟", "new": true, "role": "主角", "set": {"职位": "网格经理"}, "learns": ["T001"]}
  ],
  "foreshadow": {
    "plant":   [{"id": "F001", "desc": "...", "payoff_plan": "第2弧", "priority": "高"}],
    "advance": [{"id": "F001", "note": "..."}],
    "pay":     ["F001"],
    "drop":    [{"id": "F002", "reason": "..."}]
  },
  "truth": {
    "add":    [{"id": "T001", "fact": "作者真相，读者暂不知道"}],
    "reveal": ["T001"]
  }
}
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
CARD = ROOT / "追踪" / "上下文.md"
FILES = {
    "chapters": [],          # 逐章记录
    "characters": {},        # id -> 角色快照
    "foreshadow": {},        # id -> 伏笔
    "truth": {},             # id -> {fact, planted, revealed}
}
CHANGE = {"局面", "权力", "关系", "认知"}
HOOKS = {"危机", "悬念", "反转", "挑衅", "倒计时", "抉择"}


def load():
    return {k: json.loads((STATE / f"{k}.json").read_text(encoding="utf-8")) if (STATE / f"{k}.json").exists()
            else v for k, v in FILES.items()}


def save(st):
    STATE.mkdir(exist_ok=True)
    for k in FILES:
        (STATE / f"{k}.json").write_text(json.dumps(st[k], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(st, d):
    errs, warns = [], []
    last = st["chapters"][-1]["chapter"] if st["chapters"] else 0
    ch = d.get("chapter")
    if ch != last + 1:
        errs.append(f"章号应为 {last + 1}，收到 {ch}")
    for f in ("title", "summary", "change_type", "hook_type"):
        if not d.get(f):
            errs.append(f"缺少字段 {f}")
    if d.get("change_type") and d["change_type"] not in CHANGE:
        errs.append(f"change_type 必须是 {sorted(CHANGE)} 之一（防注水：本章必须真的改变了什么）")
    if d.get("hook_type") and d["hook_type"] not in HOOKS:
        errs.append(f"hook_type 必须是 {sorted(HOOKS)} 之一")
    recent = st["chapters"][-2:]
    if recent and d.get("hook_type") == recent[-1].get("hook_type"):
        warns.append(f"章末钩子类型与上一章相同（{d['hook_type']}），连续 2 章同类会让读者麻木")
    if len(recent) == 2 and d.get("payoff_type") not in (None, "无") and \
            all(r.get("payoff_type") == d.get("payoff_type") for r in recent):
        warns.append(f"爽点类型连续 3 章都是“{d['payoff_type']}”，请轮换")
    if len(recent) == 2 and all(r.get("payoff_type") in (None, "无") for r in recent) and d.get("payoff_type") in (None, "无"):
        warns.append("连续 3 章没有爽点")

    chars = st["characters"]
    for c in d.get("characters", []):
        if c.get("new"):
            if c["id"] in chars:
                errs.append(f"角色 {c['id']} 已存在，不能 new")
            if not c.get("name"):
                errs.append(f"新角色 {c['id']} 缺少 name")
        elif c["id"] not in chars:
            errs.append(f"未知角色 {c['id']}（新角色需 new: true）")
    truth = dict(st["truth"])
    for t in d.get("truth", {}).get("add", []):
        if t["id"] in truth:
            errs.append(f"真相 {t['id']} 已存在")
        truth[t["id"]] = t
    for tid in d.get("truth", {}).get("reveal", []):
        if tid not in truth:
            errs.append(f"揭示了不存在的真相 {tid}")
        elif st["truth"].get(tid, {}).get("revealed"):
            errs.append(f"真相 {tid} 已在第 {st['truth'][tid]['revealed']} 章揭示过")
    for c in d.get("characters", []):
        for tid in c.get("learns", []):
            if tid not in truth:
                errs.append(f"角色 {c['id']} 得知了不存在的真相 {tid}")

    fs = dict(st["foreshadow"])
    for f in d.get("foreshadow", {}).get("plant", []):
        if f["id"] in fs:
            errs.append(f"伏笔 {f['id']} 已存在")
        fs[f["id"]] = f
    for key in ("advance", "drop"):
        for f in d.get("foreshadow", {}).get(key, []):
            if f["id"] not in fs:
                errs.append(f"{key} 了不存在的伏笔 {f['id']}")
    for fid in d.get("foreshadow", {}).get("pay", []):
        if fid not in fs:
            errs.append(f"回收了不存在的伏笔 {fid}")
        elif st["foreshadow"].get(fid, {}).get("status") == "已回收":
            errs.append(f"伏笔 {fid} 已回收过")
    stale = [f for f in st["foreshadow"].values()
             if f.get("status") == "活跃" and ch and ch - f.get("last_touch", f["planted"]) >= 20
             and f["id"] not in {x["id"] for x in d.get("foreshadow", {}).get("advance", [])}
             and f["id"] not in d.get("foreshadow", {}).get("pay", [])]
    for f in stale:
        warns.append(f"伏笔 {f['id']} 已 {ch - f.get('last_touch', f['planted'])} 章没有推进：{f['desc'][:20]}")
    return errs, warns


def apply(st, d):
    ch = d["chapter"]
    st["chapters"].append({k: d.get(k) for k in
                           ("chapter", "title", "story_date", "summary", "change_type", "payoff_type", "hook_type", "events")})
    for c in d.get("characters", []):
        cur = st["characters"].setdefault(c["id"], {"id": c["id"], "name": c.get("name"), "role": c.get("role", ""),
                                                    "first": ch, "attrs": {}, "knows": []})
        cur["attrs"].update(c.get("set", {}))
        cur["last_seen"] = ch
        for tid in c.get("learns", []):
            if tid not in cur["knows"]:
                cur["knows"].append(tid)
    for t in d.get("truth", {}).get("add", []):
        st["truth"][t["id"]] = {"id": t["id"], "fact": t["fact"], "planted": ch, "revealed": None}
    for tid in d.get("truth", {}).get("reveal", []):
        st["truth"][tid]["revealed"] = ch
    fsd = d.get("foreshadow", {})
    for f in fsd.get("plant", []):
        st["foreshadow"][f["id"]] = {**f, "planted": ch, "last_touch": ch, "status": "活跃", "log": []}
    for f in fsd.get("advance", []):
        st["foreshadow"][f["id"]]["last_touch"] = ch
        st["foreshadow"][f["id"]]["log"].append(f"第{ch}章：{f.get('note', '')}")
    for fid in fsd.get("pay", []):
        st["foreshadow"][fid].update(status="已回收", paid=ch, last_touch=ch)
    for f in fsd.get("drop", []):
        st["foreshadow"][f["id"]].update(status="已废弃", reason=f.get("reason"), last_touch=ch)


def render(st):
    chs = st["chapters"]
    cur = chs[-1] if chs else None
    out = ["<!-- 由 tools/state_commit.py 生成，请勿手改 -->", "# 上下文卡", ""]
    out += ["## 当前位置",
            f"- 已完成：第 {cur['chapter']} 章《{cur['title']}》 · 故事时间 {cur.get('story_date') or '-'}" if cur else "- 尚未开写",
            ""]
    out += ["## 最近三章"] + [f"- 第{c['chapter']}章《{c['title']}》[{c['change_type']}变化 / 爽点:{c.get('payoff_type') or '无'} / 钩子:{c['hook_type']}] {c['summary']}"
                           for c in chs[-3:]] + [""]
    out += ["## 钩子与爽点轨迹（近 10 章）",
            "- 钩子：" + " → ".join(c["hook_type"] for c in chs[-10:]),
            "- 爽点：" + " → ".join(c.get("payoff_type") or "无" for c in chs[-10:]), ""]
    act = [f for f in st["foreshadow"].values() if f["status"] == "活跃"]
    out += ["## 活跃伏笔"] + [f"- {f['id']}｜{f['desc']}｜埋于第{f['planted']}章｜计划回收：{f.get('payoff_plan', '未定')}｜{f.get('priority', '')}"
                          for f in sorted(act, key=lambda x: x["id"])] + [""]
    out += ["## 角色快照"]
    for c in sorted(st["characters"].values(), key=lambda x: x["id"]):
        attrs = "；".join(f"{k}:{v}" for k, v in c["attrs"].items())
        out.append(f"- {c['id']} {c['name']}（{c['role']}）最近出场第{c.get('last_seen')}章｜{attrs}｜已知真相：{','.join(c['knows']) or '无'}")
    out += ["", "## 读者已知 / 作者真相"]
    for t in sorted(st["truth"].values(), key=lambda x: x["id"]):
        tag = f"读者已知（第{t['revealed']}章揭示）" if t["revealed"] else "⚠️ 仅作者知道，正文不可当作读者已知"
        out.append(f"- {t['id']}｜{tag}｜{t['fact']}")
    CARD.parent.mkdir(exist_ok=True)
    CARD.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "check", "commit", "render", "amend"],
                    help="amend：章节重写后修订已入账章节的记录（摘要/事件/类型），并可补登新角色与属性")
    ap.add_argument("delta", nargs="?")
    a = ap.parse_args()
    if a.cmd == "init":
        if any((STATE / f"{k}.json").exists() for k in FILES):
            sys.exit("账本已存在，拒绝覆盖")
        save({k: (v.copy() if hasattr(v, "copy") else v) for k, v in FILES.items()})
        render(load())
        print("已初始化 state/ 与 追踪/上下文.md")
        return
    st = load()
    if a.cmd == "render":
        render(st)
        print(f"已生成 {CARD}")
        return
    if not a.delta:
        sys.exit("需要 delta 文件")
    d = json.loads(Path(a.delta).read_text(encoding="utf-8"))
    if a.cmd == "amend":
        idx = next((i for i, c in enumerate(st["chapters"]) if c["chapter"] == d.get("chapter")), None)
        if idx is None:
            sys.exit(f"[error] 第 {d.get('chapter')} 章尚未入账，不能 amend")
        rec = st["chapters"][idx]
        for k in ("title", "story_date", "summary", "change_type", "payoff_type", "hook_type", "events"):
            if k in d:
                rec[k] = d[k]
        if rec["change_type"] not in CHANGE or rec["hook_type"] not in HOOKS:
            sys.exit("[error] change_type / hook_type 不合法")
        for c in d.get("characters", []):
            if c.get("new"):
                if c["id"] in st["characters"]:
                    sys.exit(f"[error] 角色 {c['id']} 已存在")
                st["characters"][c["id"]] = {"id": c["id"], "name": c["name"], "role": c.get("role", ""),
                                             "first": rec["chapter"], "last_seen": rec["chapter"], "attrs": {}, "knows": []}
            elif c["id"] not in st["characters"]:
                sys.exit(f"[error] 未知角色 {c['id']}")
            st["characters"][c["id"]]["attrs"].update(c.get("set", {}))
        save(st)
        render(st)
        print(f"第 {rec['chapter']} 章记录已修订")
        return
    errs, warns = validate(st, d)
    for w in warns:
        print(f"[warn] {w}")
    for e in errs:
        print(f"[error] {e}")
    if errs:
        sys.exit(1)
    if a.cmd == "commit":
        apply(st, d)
        save(st)
        render(st)
        print(f"第 {d['chapter']} 章已入账，上下文卡已更新")
    else:
        print("校验通过")


if __name__ == "__main__":
    main()
