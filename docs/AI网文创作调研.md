# AI 长篇网文创作：开源项目与流程调研

> 调研日期：2026-11-02
> 目标：为后续「AI 主导创作一部能在番茄小说跑起来的长篇网文」选定可参考的方法和工作流。
> 来源：GitHub 高星项目的 README 和技能文件、中文社区文章（经搜索引擎摘要）、平台新闻报道。知乎、掘金、腾讯云等站点在本环境网络中无法直连，相关内容只能读到搜索摘要，没法逐字核对原文；X 上也没搜到质量高的相关帖子。

---

## 0. 先说结论：两条平台硬约束会决定整个方案

1. **番茄在严打「纯 AI 生成」。** 2026 年 5 月的低质内容治理公告把「滥用 AI 批量生成、缺乏原创性」归入「粗制滥造」，并配合 AI 检测和行为数据交叉判定。媒体报道有十万本量级的作品被拒签或处置，部分账号被永久封禁。作家助手从 2025 年 9 月起有「是否使用 AI」的强制勾选，依据是《人工智能生成合成内容标识办法》。
   → **百分之百无人介入的「全自主」路线，合规风险和被检测风险都很高。** 比较现实的定位是：AI 承担绝大部分生产，作者在**选题、核心设定、关键剧情节点、终审**这几个环节真正拍板和改写，并如实勾选 AI 声明。坊间说的「至少 30% 人工加工」没有找到官方原文，只能当参考。
2. **新人只能「完本签约」（2026-10-12 起）。** Lv.0 和 Lv.1 作者的新书必须写完，并达到 15 万或 30 万字才能申请签约。审核是整书一次性审，签约后内容不可修改或删除。以前那种「2 万字先签、边写边看数据」的打法，新号已经用不了。
   → 第一本书应该**按完本来规划**：先定 15 万或 30 万字档，整书写完、全书审校之后再提交。这反而很适合 AI 流程，因为可以整书生成、整书回改，不受日更压力影响。

---

## 1. 开源项目全景（按参考价值排序）

| 项目 | Stars | 形态 | 核心思路 | 对我们的价值 |
|---|---|---|---|---|
| [zenstory-ai/oh-story-claudecode](https://github.com/zenstory-ai/oh-story-claudecode) | ~8.2k | Claude Code / Codex Skill 包（14 个） | 扫榜 → 拆文 → 大纲 → 正文 → 去 AI 味 → 封面 → 番茄发布资料；用文件系统当记忆；写前、写后都有确定性门禁脚本 | ★★★★★ 最贴近「Claude Code + 番茄」，可以直接借用 |
| [lingfengQAQ/webnovel-writer](https://github.com/lingfengQAQ/webnovel-writer) | ~7.8k | Claude Code 插件（v6） | 「防遗忘、防幻觉」，号称支持 200 万字；章节合同 → 写 → 审 → 事实提取 → CHAPTER_COMMIT → 投影到 state/index/RAG | ★★★★★ 长篇一致性系统设计最完整 |
| [Narcooo/inkos](https://github.com/Narcooo/inkos) | ~11k | Node CLI / Studio | 多 Agent 分工：Radar（趋势）、Planner（本章意图 must-keep / must-avoid）、Composer（上下文编排）、Writer、Settler（状态增量）、Auditor、Reviser；状态用 JSON 作为权威源，Markdown 只是投影 | ★★★★ Agent 角色划分值得照抄 |
| [Niger12138/novel-outline-craft](https://github.com/Niger12138/novel-outline-craft) | — | Skill | 四级结构：全书 → 人物 → 卷纲 → 逐章细纲；带**番茄档位**硬指标；结构层去 AI 味的机检 | ★★★★ 细纲模板和番茄参数可以直接用 |
| [LAY-lgtm/novel-writing-framework](https://github.com/LAY-lgtm/novel-writing-framework) | ~270 | 方法论 Skill | 从 500+ 章实战提炼；番茄专项「写前阻断五问」「三章一轮扫描」；AI 指纹量化指标 | ★★★★ 番茄爽点和节奏规则最具体 |
| [NousResearch/autonovel](https://github.com/NousResearch/autonovel) | ~1.7k | Python 管线 | 借鉴 autoresearch 的「修改 → 评估 → 保留或丢弃」循环；先打分门槛，再做对抗式删改、读者小组评审、Opus 双人格审稿 | ★★★ 自动评估和迭代回改的机制可借鉴（英文文学向） |
| [PenglongHuang/chinese-novelist-skill](https://github.com/PenglongHuang/chinese-novelist-skill) | ~4.5k | Skill | 三层问答 → 规划确认 → 全自动逐章写（可开子 Agent 并行）→ 自动校验修复 | ★★★ 偏 10–50 章中篇，流程轻 |
| [ExplosiveCoderflome/AI-Novel-Writing-Assistant](https://github.com/ExplosiveCoderflome/AI-Novel-Writing-Assistant) | ~3.3k | Web 应用（LangGraph） | 世界观、写法引擎、RAG、「自动导演」 | ★★ 产品化程度高，我们用不上 |
| [YILING0013/AI_NovelGenerator](https://github.com/YILING0013/AI_NovelGenerator) | ~6.4k | 本地应用 | 卷 → 弧 → 场景 → 章节的分层规划 | ★★ 分层思路可以参考 |
| [UnderleveledBuilder/awesome-novel-skills](https://github.com/UnderleveledBuilder/awesome-novel-skills) | — | 索引 | 收录约 50 个网文 Skill，并对选题能力做了横评 | 索引，后续查漏补缺用 |

社区实战案例（只读到摘要）：
- 腾讯云开发者社区《用 Claude Code 写了一部 70 万字的小说》：先让 Claude 扫番茄热门、拆同题材爆款，再生成题材提示词；大纲靠一问一答共同推出来；**「一上来让它自由写，必崩」**；拆成主笔、编辑（节奏和爽点）、设定（一致性）、读者模拟、去 AI 味几个 Agent。
- 掘金《用 AI 写了 78 万字网文，结果出乎意料》：320 章、7 卷都市种田文，全程零人工。主要翻车点是 **AI 不知道写什么时就注水**，比如把「适量浇水」拆成七八条重复短句。
- Reddit 讨论一部约 30 万词、由 Claude 写的小说：读起来仍然明显是 AI 写的，超过约 10 万 token 后连贯性会下降。

---

## 2. 各家共同的方法论（提炼）

### 2.1 市场先行：扫榜 → 拆文 → 定题材
- 先看平台榜单和题材趋势，再拆 2–3 本同题材爆款：黄金三章、爽点分布、钩子类型、金手指设计。
- 选材公式：**「套路 = 确定性的情绪满足」**（oh-story 的说法）。新意放在一句话卖点上，结构用成熟套路。

### 2.2 结构分层，逐级确认，不跳级
```
L1 全书大纲（卖点 / 故事引擎 / 卷数 / 结局）
 └ L2 人物（动机、说话方式、成长弧、角色语音卡）
    └ L3 卷纲（本卷目标、起伏、卷末兑现）
       └ L4 逐章细纲（阻碍、本章变化、爽点、章末钩子、伏笔埋收）
```
- 下游写不下去就回上游改，不在细纲上打补丁。
- 「本章变化」必须落在**局面、权力、关系、认知**四类之一，这样能防止注水章。

### 2.3 文件即记忆，状态有权威源
- 设定、大纲、正文、追踪分目录落盘。上下文被压缩或换会话后，从文件恢复进度。
- 权威状态用结构化 JSON（角色状态、伏笔台账、时间线、物品和战力），Markdown 由它生成，供人阅读。
- **「作者真相」和「读者已知」分开记录**，用来检查角色有没有提前知道答案、伏笔是不是已经揭示。
- 每章写前，由编排 Agent 只挑出与本章相关的状态，生成「写作任务书」交给写手，不要把全部设定都塞进上下文。

### 2.4 单章流水线（带门禁）
```
写前检查（细纲齐全？追踪已同步？）
 → 生成本章任务书（must-keep / must-avoid）
 → 写手起草
 → 审稿 Agent（一致性 / OOC / 节奏 / 爽点 / 追读力）——有 blocking 问题就打回
 → 去 AI 味（脚本 lint + 读感修订）
 → 事实提取 → 状态提交（伏笔、角色、时间线增量）
 → 备份
```

### 2.5 去 AI 味：用机检，不靠「请写得像人」
大家反复提到的 AI 指纹：
- **「不是 X，而是 Y」** 句式（有人一本书里出现 663 次，清洗到 31 次花了几十轮），目标是每章不超过 2 次。
- 破折号泛滥（叙事里的破折号用脚本批量替换）、「没有犹豫，没有生涩」这种排比否定、「声音不大，却……」。
- 口水词：缓缓、微微、不禁、深吸一口气、眼底闪过、心中升起、意味深长、空气中弥漫、命运的齿轮……
- 结构层面：每章收束方式相同、群像轮流点名、抒情式结尾、所有角色一个腔调。
- 正文里出现元话语（「第 X 章」「前文」「本卷」）。
- **三章一轮扫描**：每写 3 章就跑一遍指纹统计，不要攒到几十章再修，后期修的成本是初稿的 10 倍以上。

### 2.6 评估与迭代（autonovel 的思路）
- 每章打分，达到门槛才保留，否则重写。
- 用「删 500 字」做对抗式编辑，找出注水部分。
- 多人格读者小组（小白读者、老书虫、编辑、毒舌评论）做整卷评审。
- 两章之间做 Elo 对比，挑出更好的版本。

---

## 3. 番茄平台写作参数（社区经验，不是官方标准）

| 维度 | 番茄档建议 |
|---|---|
| 单章字数 | 2000–3000（不低于约 1800） |
| 开篇 | 前 300 字必须出冲突或物理事件；第 1 章亮出金手指；禁止用世界观介绍开篇 |
| 爽点密度 | 3 章一小爽、5–10 章一大爽、约 20 章一次重大反转；高潮后的日常不超过 2 章 |
| 爽点类型 | 打脸、碾压、装逼、截胡、复仇、升级、反转，并且轮换使用；「理解、和解、放下」这类文青爽不算 |
| 爽点自检 | 能用「谁 + 动词 + 谁 + 损失」描述 |
| 章末钩子 | 每章都要有；同类型不连续 2 章；常用危机突降、悬念反转、挑衅叫板、倒计时 |
| 语言 | 短句、短段、单句不超过 25 字左右；对话要能「读出来」，因为番茄有大量听书用户 |
| 反派 | 前 1–2 章就要造成可见伤害；每 10 章至少 1 次让读者觉得主角会输的生死局 |
| 主角 | 不能当旁观者；每 10 章至少亲手解决一次 |
| 追更衰减点 | 社区说法是第 7、21、45 章，这些位置要放不能跳过的内容 |
| 包装 | 书名带搜索热词，一眼看出题材；简介第一句给冲突，3–5 句交代困境、金手指和第一阶段目标；封面一眼能看出品类 |

> 「追读率 X% 生死线」这类数字多数没有官方来源，而且有作者说番茄后台已经取消了追读率指标。参考上表的节奏，不要迷信具体数值。

---

## 4. 给我们这部书的推荐流程（草案）

结合以上调研，推荐以 **oh-story 的流程骨架 + webnovel-writer / InkOS 的状态系统 + 番茄档硬指标 + autonovel 的评估回改循环** 为蓝本，在本仓库里用 Claude Code 自己搭一套轻量版。

**阶段 A：选题（需要你拍板）**
1. 扫番茄当前榜单的题材趋势，拆 2–3 本同题材爆款的前 10 章。
2. 出 3 个候选选题，每个包含一句话卖点、金手指、核心冲突、目标读者（男频或女频）、对标书。
3. 你选一个，或者修改。

**阶段 B：设定与结构（需要你确认 L1 和 L2）**
4. L1 全书大纲：按完本 30 万字左右规划（约 120 章 × 2500 字，3–4 卷），结局先定好。
5. L2 人物卡和角色语音卡；L3 卷纲；L4 第一卷逐章细纲，按番茄 9 字段模板。
6. 建立 `state/` 状态文件：角色、伏笔台账、时间线、作者真相和读者已知。

**阶段 C：生产（AI 主导，三章一轮）**
7. 按章执行：任务书 → 起草 → 审稿 → 去 AI 味 lint → 状态提交。
8. 每 3 章：指纹统计（不是 X 是 Y、破折号、口水词、钩子类型重复）。
9. 每 10 章：爽点和钩子分布表、读者小组评审，不达标就回改。
10. 每卷结束：全卷一致性审查，再展开下一卷细纲。

**阶段 D：整书打磨与提交（需要你终审）**
11. 黄金三章单独精修多轮，可以用多版本 Elo 对比。
12. 全书去 AI 味复扫，你通读关键章节并亲手改写。
13. 准备书名、简介、标签和封面提示词，如实勾选 AI 声明，按完本签约提交。

**目录结构草案**
```
novel/
├── 00_市场/        扫榜、拆文报告、选题卡
├── 01_设定/        全书大纲、世界观、人物卡、语音卡
├── 02_大纲/        卷纲、逐章细纲
├── 03_正文/        第001章_xxx.md …
├── state/          characters.json  foreshadow.json  timeline.json  truth_vs_known.json
├── reviews/        单章审稿、十章报告、卷审
└── tools/          ai_fingerprint.py（指纹扫描）  wordcount.py  outline_check.py
```

---

## 5. 风险与预期

- **能不能「火」没法保证。** 番茄头部作品极少，流量由算法赛马决定。现实的第一目标是：完本、通过签约审核、首秀数据不扑街。
- **AI 检测和平台政策会继续收紧。** 人工深度参与的环节越多，被判定为「粗制滥造」的风险越低。
- **中后段注水和连贯性漂移**是所有案例共同的翻车点，所以状态系统和「本章变化」硬约束是必需品，不是可选项。
- **AI 文风同质化**：需要你提供或认可一份文风样本（对标书片段），作为 voice 基准。

---

## 参考来源

- GitHub：[oh-story-claudecode](https://github.com/zenstory-ai/oh-story-claudecode) · [webnovel-writer](https://github.com/lingfengQAQ/webnovel-writer) · [inkos](https://github.com/Narcooo/inkos) · [novel-outline-craft](https://github.com/Niger12138/novel-outline-craft) · [novel-writing-framework](https://github.com/LAY-lgtm/novel-writing-framework)（含 `skills/piqie-writing.md` 番茄专项） · [autonovel](https://github.com/NousResearch/autonovel) · [chinese-novelist-skill](https://github.com/PenglongHuang/chinese-novelist-skill) · [AI-Novel-Writing-Assistant](https://github.com/ExplosiveCoderflome/AI-Novel-Writing-Assistant) · [AI_NovelGenerator](https://github.com/YILING0013/AI_NovelGenerator) · [awesome-novel-skills](https://github.com/UnderleveledBuilder/awesome-novel-skills)
- 实战文章：[腾讯云：用 Claude Code 写 70 万字小说](https://cloud.tencent.com/developer/article/2665477) · [掘金：AI 自己写 78 万字网文](https://juejin.cn/post/7648082945085980718) · [AI 写小说工作流三模型分工](https://maplefeather.com/article/ai-novel-writing-workflow-gpt-claude-gemini-2026)
- 平台政策：[番茄 2026 上半年十大变化](https://www.toutiao.com/article/7651950505349988874/) · [番茄封禁金番作者的标准](https://k.sina.com.cn/article_7879848900_1d5acf3c406803cus0.html) · [钛媒体：AI 小说大逃杀](https://www.tmtpost.com/8000672.html) · [界面·电厂：番茄调整签约规则](https://m.jiemian.com/article/15208321.html) · [人人都是产品经理：签约新规解读](https://www.woshipm.com/operate/6480946.html)
- 研究：[Towards Human-Level Book-Writing Capability (arXiv 2605.17064)](https://arxiv.org/pdf/2605.17064)
