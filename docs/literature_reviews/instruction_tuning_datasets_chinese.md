# Chinese (zh) Instruction-Tuning / SFT / Post-Training Datasets — A Literature Review
# 中文指令微调 / SFT / 后训练数据集综述

## ⭐ Dataset Quality Ranking — Hands-On Deep Sampling / 数据集质量实测排名（深度抽样）

*Method / 方法:* Every ranked dataset below was **actually opened and read**, not summarized from its card. Rows were pulled directly from each repo's auto-converted **Parquet shards on the HuggingFace CDN via HTTP range requests** (a `RangeFile` + `pyarrow.ParquetFile` reader, so only the needed row groups were downloaded), at **offsets 0 / 2,000 / 20,000** within the first shard, plus a **last-shard deep probe** (middle and final row group) for the top candidates. Row counts quoted are the **real `num_rows` from the Parquet metadata or the `datasets-server` `/size` endpoint**, not card claims — where they disagree, both are shown. Judged on: Chinese fluency and naturalness, **language purity** (is it natively Chinese or translationese / actually English?), answer depth, formatting noise, template/boilerplate duplication, factual soundness, and **deep-offset degradation**. Gated / unreadable repos are listed separately at the bottom and are **not ranked**.
/ 下列每个数据集均**实际打开并逐条阅读**，而非依据数据卡片转述。行数据通过 **HTTP Range 请求直读 HuggingFace CDN 上自动转换的 Parquet 分片**（`RangeFile` + `pyarrow.ParquetFile`，仅下载所需 row group），采样位置为首个分片的 **offset 0 / 2,000 / 20,000**，并对头部候选追加**末尾分片深探**（中间与最后一个 row group）。所引行数均取自 **Parquet 元数据或 `datasets-server` `/size` 的真实 `num_rows`**，而非卡片声称值；两者不一致时并列给出。评判维度：中文流畅度与自然度、**语言纯度**（原生中文还是翻译腔／实为英文）、答案深度、格式噪声、模板／样板重复、事实正确性，以及**深层偏移是否退化**。受限或无法读取的库单列于末尾，**不参与排名**。

---

### Tier 1 — Excellent / 优秀

**1. `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` — 110,000 rows · Apache-2.0 · Tier: Excellent / 优秀.**
Sampled offsets 0 / 2,000 / 20,000 **and** the last shard (row groups 14 and 28 of 29) — quality is uniform end to end. The single best *native-Chinese reasoning* SFT set I read: every row carries a separate `reasoning_content` (R1's CoT) and `content` (final answer), plus a `repo_name` provenance field and a Qwen2.5-72B `score`, so you can threshold on quality before training. Answers are long, structured, and idiomatic — no translationese. | 采样 offset 0/2,000/20,000 **并深探末尾分片**（第 29 个分片的第 14、28 个 row group），全程质量均匀。是我读到的**最好的原生中文推理 SFT 数据**：每行分离 `reasoning_content`（R1 思维链）与 `content`（最终答案），并带 `repo_name` 溯源字段与 Qwen2.5-72B 的 `score`，可在训练前按分数过滤。答案长、结构清晰、地道，无翻译腔。
> `input`: 能给我讲一个寓意深刻的故事吗？ → `reasoning_content`: 好的，用户让我讲一个寓意深刻的故事。首先，我需要确定用户的需求是什么。他们可能想要一个能引发思考、有深层含义的故事…
> `input`: 达斯魔在《星球大战：义军起义》中的谢幕有何特别之处？ (last-shard probe / 末尾分片)
> **Deep-offset:** Stable. **Caveat:** the math slice is prompt-translated — a last-shard row carries `repo_name: meta-math/GSM8K_zh` and reads "Courtney喜欢收集弹珠…" (an English word problem in Chinese clothing). The general/STEM slice is natively Chinese. | 深层稳定。**注意**：数学部分为翻译提示（末尾分片一行 `repo_name` 为 `meta-math/GSM8K_zh`，内容是"Courtney喜欢收集弹珠…"这类英文应用题的中译）；通用与 STEM 部分为原生中文。

**2. `allenai/WildChat-1M` — 837,989 conversations, of which **23.27% are Chinese** · ODC-BY · Tier: Excellent / 优秀.**
I counted the full `language` column over **3 of 14 shards = 179,571 rows**: English 85,660 (47.70%), **Chinese 41,787 (23.27%)**, Russian 21,422 (11.93%), French 4,315 (2.40%), Spanish 4,165 (2.32%). Extrapolated, that is on the order of **~195,000 Chinese conversations** — making WildChat one of the largest sources of *genuine Chinese user prompts* in existence, and the only one in this review where the prompt distribution was not invented by a model or a translator. I then pulled real Chinese rows out of a deep shard (shard 8/14, row group 6/60) to check quality: they carry `country`, `state`, `language`, `turn`, `toxic`, `redacted` and full `openai_moderation` / `detoxify_moderation` fields, so you can filter precisely. | 我对 **14 个分片中的 3 个、共 179,571 行**的 `language` 列做了全量统计：英语 85,660（47.70%）、**汉语 41,787（23.27%）**、俄语 21,422（11.93%）、法语 4,315、西班牙语 4,165。按比例外推约 **19.5 万条中文对话**——是现存最大的**真实中文用户提问**来源之一，也是本综述中唯一提示分布不由模型或译者虚构的数据。随后从深层分片（第 8/14 个分片、第 6/60 个 row group）取真实中文行验证质量，每行带国家/省份/语言/轮次/毒性/脱敏标记与完整审核元数据，可精确过滤。
> (shard 8, `country: China`, `state: Zhejiang`) 除了猫狗，在家值得养的宠物有哪些，可以互动的 → 除了猫狗之外，还有许多宠物可以选择养在家里…1. 兔子：兔子是温和友善的宠物…2. 鼠类宠物（如仓鼠、倭鼠等）…
> (shard 8, `country: Hong Kong`) 大中小学思政课优质课程资源共建共享的现实策略有哪些，请写对策不少于500字 → 1. 完善资源管理机制…2. 加强教师培训…
> **Deep-offset:** Chinese rows at shard 8 are the same quality as anywhere else. **Caveats:** responses are `gpt-3.5-turbo-0613` / `gpt-4` outputs (OpenAI ToU exposure), the prompt mix includes a lot of homework/essay-mill and NSFW roleplay requests, and it is a *log*, not a curated set — you must filter on `toxic`, `redacted`, `turn`, and length. | 第 8 个分片的中文行与他处质量一致。**注意**：回答为 `gpt-3.5-turbo-0613`/`gpt-4` 输出（涉 OpenAI 使用条款）；提示中大量作业代写与 NSFW 角色扮演请求；它是**日志**而非策展数据集，必须按 `toxic`/`redacted`/`turn`/长度过滤。

**3. `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese` — 200,000 rows (2 shards × 100,000) · no license tag · Tier: Excellent / 优秀.**
Sampled offsets 0 / 2,000 / 20,000 in shard 1 and row groups 50 / 99 of shard 2. Self-synthesized from `Qwen/Qwen2-72B-Instruct` with nothing but the chat template as a prompt, so the Chinese is **natively generated, not translated** — questions read like real Chinese users ("请输出所有跟政企市场相关的关键词列表"). Every row ships an unusually rich filtering harness: `instruct_reward` (FsfairX-LLaMA3-RM), `input_quality`, `difficulty`, `task_category`, `llama_guard_2`, `min_neighbor_distance`, `repeat_count`. That metadata is what makes it the most *trainable* zh set here — you can cut the bottom by reward and dedup by neighbour distance without extra work. | 在第 1 个分片采样 offset 0/2,000/20,000，并在第 2 个分片采样第 50、99 个 row group。以聊天模板为唯一提示，从 `Qwen2-72B-Instruct` 自合成，中文为**原生生成而非翻译**，提问口吻贴近真实中文用户。每行附带极完备的过滤元数据（奖励分、输入质量、难度、任务类别、安全标签、近邻距离、重复计数），无需额外工作即可按奖励截断、按近邻去重——这是它在本表中"最可训练"的原因。
> 设计一个算法，可以识别人脸上的眼镜。 → 1. 数据集准备：首先，需要一个包含大量带眼镜和不带眼镜的人脸图像的数据集…（last-shard probe / 末尾分片）
> **Deep-offset:** No degradation across both shards. Minor defect: occasional garbled characters in synthesized prompts — one row reads "平衡因子擯至2" (擯 is nonsense there). | 两个分片均无退化。小瑕疵：合成提问偶有乱字，如"平衡因子擯至2"（"擯"在此处不成词）。

**4. `m-a-p/COIG-CQIA` — 44,694 rows across 13 configs (card claims 48,384) · **no license declared** · Tier: Excellent (small) / 优秀（规模小）.**
Sampled the `ruozhiba`, `zhihu`, `exam`, `wiki` and `xhs` configs. The only set here whose answers are predominantly **human-written and human-verified** (each row has `answer_from: human|llm` and `human_verified: true|false`), sourced from Zhihu, Douban, Xiaohongshu, SegmentFault, MBA智库, exams and encyclopedias. Per-example naturalness is the highest of anything I sampled — real Chinese register, real Chinese subject matter, LaTeX and Markdown preserved. | 唯一一个答案以**人写并经人工校验**为主的数据集（每行带 `answer_from` 与 `human_verified` 字段），来源为知乎、豆瓣、小红书、SegmentFault、MBA智库、考试与百科。单条自然度为全表最高：真实的中文语域、真实的中文题材，保留 LaTeX 与 Markdown。
> `zhihu`: 求一个整数的所有素数因子的思路是什么？ → 1. 一般的小数可以用简单筛法找出质数列表…2. 再大一点的数 $$n$$ 就用Pollard的 $$\rho$$ 算法…（`answer_from: human`, `human_verified: true`）
> `ruozhiba`: 石油也是油，为啥没人用它来炒菜？ → 石油虽然也是一种油，但它并不适用于烹饪…1. **化学成分**…2. **毒性**…
> **Deep-offset:** Not applicable at this scale (largest config is 11,288 rows), but quality is **uneven across configs, not across offsets**: the `xhs` (Xiaohongshu) config is emoji-saturated giveaway marketing — "免费送！爱麦士行李箱实力宠粉！[偷笑R] ✨这款奶白行李箱颜值超高…👉8月27日下午3点小红书官方平台自动开奖" — and should be dropped or heavily downweighted. Filter on `human_verified == true`. | 规模小无需深探，但**质量差异体现在子集之间而非偏移之间**：`xhs`（小红书）子集为表情符号密集的抽奖营销文案，应剔除或大幅降权；建议按 `human_verified == true` 过滤。

**5. `Mxode/Chinese-Instruct` — 4,845,389 rows across 14 configs · CC-BY-SA-4.0 · Tier: Excellent (aggregate) / 优秀（聚合）.**
Sampled the `magpie`, `infinity-instruct`, `firefly`, `stem_zh` and `dpsk-r1-distil` configs at offsets 0 / 2,000 / 20,000. **This is the single most practical starting point for a zh SFT mix in 2025-26**: it is a curated, language-detected, re-answered aggregation of the whole open zh ecosystem, with a declared license (CC-BY-SA-4.0) — which most of its sources lack. Verified per-config row counts: `firefly` 1,061,823 · `chinese-medical` 998,387 · `Chinese-QA-AFAF` 676,685 · `magpie` 542,044 · `infinity-instruct` 386,426 (the zh single-turn slice of Infinity-Instruct, re-language-detected because "其自带的语言元数据仍有误") · `industryinstruction` 358,582 · `stem_zh` 214,646 · `sof-c-zh` 198,335 · `chinese-reasoning-distil` 179,037 · `dpsk-r1-distil` 100,331 · `neo_sft_phase2` 58,549 · `disc-law` 51,463 · `coig-cqia` 10,306 · `psycho-10k-dpsk-r1` 8,775. | 在 `magpie`/`infinity-instruct`/`firefly`/`stem_zh`/`dpsk-r1-distil` 五个子集上采样 offset 0/2,000/20,000。**这是 2025–26 年做中文 SFT 混合最实用的单一起点**：对整个开源中文生态做了语言检测、重答与筛选的聚合，且声明了许可（CC-BY-SA-4.0）——而它的多数上游并没有。各子集真实行数见上。
> `stem_zh`: 请解释一下中草药黄连的药理学作用。 → 1. **抗菌作用**：黄连中的主要活性成分黄连素（Berberine）具有强大的抗菌活性…
> `firefly`: 修订文本：…28位来自巴基斯坦（巴）的国家议员一同签署了一项议案… → 修订后的文本如下：5月28日，巴基斯坦的议会内发生了一个重要事件…
> **Deep-offset:** Stable. **Caveat — the `magpie` config is the weak link:** its instructions were GPT-4o-rewritten from English Magpie and read as translationese with non-Chinese context ("在新加坡，一家只有单一所有者的公司仍然需要向IRAS提交税务申报表"), and at offset 20,000 a row asks about "我妹妹" but the answer replies about "你姐姐" — a sister/elder-sister slip that only happens when translating from English `sister`. Prefer `stem_zh`, `firefly`, `dpsk-r1-distil`, `coig-cqia` over `magpie`. | 深层稳定。**注意——`magpie` 子集是短板**：其指令由 GPT-4o 从英文 Magpie 改写而来，翻译腔明显且语境非中文（新加坡 IRAS 报税）；offset 20,000 处一行提问"我妹妹"而回答称"你姐姐"，这是英文 `sister` 翻译才会出现的错位。优先取 `stem_zh`/`firefly`/`dpsk-r1-distil`/`coig-cqia`，慎用 `magpie`。

---

### Tier 2 — Good / 良好

**6. `BelleGroup/train_3.5M_CN` — 3,606,402 rows · **GPL-3.0 tag but research-only prose** · Tier: Good / 良好.**
Sampled offsets 0 / 2,000 / 20,000. The largest *chat-formatted* zh set that is actually readable: ShareGPT-style `conversations`, real multi-turn (sampled mean ≈2.7 messages), broad topical coverage (fitness, cooking, poetry, code, history, role-play). Fluent and clean. | 可实际读取的最大**对话式**中文数据：ShareGPT 风格 `conversations`，真实多轮（抽样均值约 2.7 条消息），题材广（健身、烹饪、诗歌、代码、历史、角色扮演），行文流畅干净。
> 写一个关于秋天的诗吧。 → 红黄绿交织，田野金黄色。风吹枫叶落，人在秋天慢步行。
> **Deep-offset:** Fluency holds, but **instruction-following does not**. At offset 20,000 a row ends "…总结他们在聊什么。" (summarize what they are discussing) and the assistant instead *continues the dialogue* ("Jia: 我觉得这次跟Ming的聊天非常有收获…"). This is the classic `gpt-3.5-turbo` self-instruct failure mode and it is present at depth, not just at the head. | 流畅度保持，但**指令遵循不保持**：offset 20,000 处指令要求"总结他们在聊什么"，助手却继续续写对话。这是 `gpt-3.5-turbo` self-instruct 的典型失败模式，且出现在深层而非仅开头。

**7. `YeungNLP/firefly-train-1.1M` — 1,649,399 rows (card says 1.15M) · **no license** · Tier: Good / 良好.**
Sampled offsets 0 / 2,000 / 20,000. Uniquely valuable for one reason: it is **not GPT-distilled**. Human-written instruction templates were applied over 23 existing Chinese NLP datasets, and it is the only large set here containing genuinely Chinese-cultural task types — 对联 (couplets), 古诗续写 (classical poetry continuation), 文言文翻译, 金庸小说 — which is directly relevant to a figurative-language project. | 之所以独特，只因一点：**非 GPT 蒸馏**。人工书写指令模板套用于 23 个既有中文 NLP 数据集，也是本表中唯一大规模包含真正中华文化任务类型（对联、古诗续写、文言文翻译、金庸小说）的数据——对研究比喻性语言的项目尤其相关。
> `kind: Couplet` — 上联：把酒邀春，春日三人醉 下联： → 梳妆佩玉，玉王点一娇
> `kind: AncientPoem` — 模仿王之道，续写古诗：浣溪沙 其十一 春日 → 水外山光淡欲无。堤边草色翠如铺…
> **Deep-offset:** Fully stable — the task mixture at offset 20,000 looks like the mixture at offset 0. **But targets are short and extractive** (NLI labels, keyword lists, one-line summaries): this trains task-following, not assistant behaviour. Use it as a *diversity* component, never as the backbone. | 深层完全稳定，offset 20,000 与 offset 0 的任务分布一致。**但目标输出短且抽取式**（NLI 标签、关键词表、单句摘要）：它训练的是任务遵循而非助手行为，只宜作**多样性**成分，不可作主干。

**8. `wangrui6/Zhihu-KOL` — 1,006,218 rows · **no license (scraped Zhihu content)** · Tier: Good, degrading / 良好但递减.**
Sampled offsets 0 / 2,000 / 20,000. Real human Chinese writing at scale, with `upvotes` and `answer_creation_time` metadata — the most authentic *voice* in this whole review, full of colloquialism, humour and internet register. | 大规模真人中文写作，附 `upvotes` 与创建时间元数据——全篇综述中语气最真实的语料，口语、幽默与网络语域丰富。
> 怎么说服男朋友买烤箱？ → emmmmm，首先想说的是，我买厨房用品一般是不用「说服」的…但是！！你还是可以告诉他烤箱的作用是可以烤制各种肉类！！！
> **Deep-offset: this one genuinely degrades.** At offset 0 answers are long and considered; at offset 20,000 I hit "C语言中，for 和 while 在汇编上有什么区别？ → while在理论上会比for快一些，详情记不太清了" — a one-line, hedging, factually unsupported answer. Must be thresholded on `METADATA.upvotes` and length before use. | **该集确实存在深层退化**：offset 0 的答案长而有据，offset 20,000 处出现"while在理论上会比for快一些，详情记不太清了"这类一句话、含糊、无依据的回答。使用前必须按 `METADATA.upvotes` 与长度过滤。

**9. `BAAI/COIG` — 178,246 rows in the `Default` split (card total 190,251; 275,985 across configs) · Apache-2.0 · Tier: Good / 良好.**
Sampled offsets 0 and 2,000. The sampled rows are dominated by the **counterfactual-correction multi-round chat (CCMC)** subset, which is structurally unusual and genuinely useful: a student asserts something false and the teacher corrects it over ~5 rounds, grounded in CN-DBpedia. This is the only open zh set that explicitly trains *disagreeing with a wrong premise* — a known weakness of sycophantic SFT models. | 采样行以**反事实修正多轮对话（CCMC）**子集为主：学生提出错误断言、教师基于 CN-DBpedia 多轮纠正。这是唯一显式训练**反驳错误前提**的开源中文数据，而这正是谄媚型 SFT 模型的已知短板。
> 玉露香梨和库尔勒香梨是同一种梨吗？ → 不是同一种梨，玉露香梨是以库尔勒香梨为母本、雪花梨为父本杂交育成的新品种。 / 玉露香梨难道不是口感多汁、味道甜美吗？ → 确实，玉露香梨以汁多、酥脆、含糖高等品质而著名，但这些品质特点并不表明它与库尔勒香梨为同一种梨。
> **Deep-offset:** Offset 0 and 2,000 are indistinguishable in kind and quality. Note the schema oddity: the `instruction` field is empty for CCMC rows and the content lives in `conversations` as `{question, answer}` pairs. | offset 0 与 2,000 在类型与质量上无差别。注意 schema 特点：CCMC 行的 `instruction` 为空，内容在 `conversations` 中以 `{question, answer}` 形式存放。

**10. `BAAI/COIG-PC-Lite` — 1,078,563 rows across splits (`train` split 216,691) · license "unknown", **non-commercial gate on the parent** · Tier: Good (task data) / 良好（任务型）.**
Sampled offsets 0 / 2,000 / 20,000 of `train`. 3,339 normalized Chinese NLP tasks with `task_type` major/minor, `domain` and source `filename` per row. Cleanly formatted and completely unambiguous, but it is task data, not chat. | 3,339 个归一化中文 NLP 任务，每行带 `task_type`（大类/小类）、`domain` 与来源文件名，格式干净、指令明确；但属任务型数据而非对话。
> 当给一个没有标注的中文文本添加实体标记时…请按照如上指示标记下面文本： → {{product_name:浙江在线杭州}}{{time:4月25日}}讯（记者{{person_name: 施宇翔}}…
> **Deep-offset:** Stable; at offset 20,000 the sampled task is Chinese-culture book identification over unpunctuated Traditional-Chinese classical text ("武王勝殷殺受立武庚以箕子歸作洪範…" → 尚書) — evidence the deep tail carries genuinely Chinese-specific tasks, not just NER. | 深层稳定；offset 20,000 处任务为对无标点繁体文言文做典籍归属判断（→ 尚書），说明深层确实含有中文专有任务，而不止于 NER。

**11. `m-a-p/COIG-Kun` — 367,592 rows (`skypile` split 57,339 · `wanjuan` · `wudao`) · **no dataset license** · Tier: Good / 良好.**
Sampled the `skypile` split. Instruction **back-translation** (Humpback-style) over Wudao / WanJuan / SkyPile, with a Yi-based label model — so it is one of the very few large zh sets with **no OpenAI model anywhere in the pipeline**. Answers are long and information-dense. | 在 Wudao/WanJuan/SkyPile 上做指令**回译**（Humpback 式），标注模型基于 Yi——因此是极少数**全流程无 OpenAI 模型**的大规模中文数据。答案长、信息密度高。
> 请提供安徽成人高考专升本护理学专业的培养目标和专业核心课程。 → 培养目标：…专业核心课程：生理学、病理生理学、基础护理学、健康评估、药理学…
> **Deep-offset:** Not probed beyond the head of `skypile`. Visible defect: hedging boilerplate tails — "请注意，上述信息可能不是最新的，建议直接咨询…" followed by a 5-item list of ways to contact a provincial education bureau. Strip these tails. | 未对 `skypile` 之外深探。可见缺陷：结尾常挂免责样板（"请注意，上述信息可能不是最新的，建议直接咨询…"）并罗列五条联系方式，建议截除。

**12. `TigerResearch/sft_zh` — 530,705 rows · Apache-2.0 · Tier: Good / 良好.**
Sampled offsets 0 / 2,000 / 20,000. ~94% is `tigerbot-alpaca-zh-0.5m`. Clean and consistent, but a striking share of the head is **Chinese-instruction-over-English-content** translation exercises. | 约 94% 来自 `tigerbot-alpaca-zh-0.5m`。干净一致，但开头有相当比例是**中文指令＋英文正文**的翻译练习。
> 将以下短文中的所有"is"动词改为"was"。 → The cake was delicious and the pie was too…
> **Deep-offset:** Improves with depth — offset 2,000 and 20,000 are fully Chinese and more assistant-like (summarization, punctuation restoration, book reviews). | 深层反而更好：offset 2,000 与 20,000 为纯中文且更接近助手行为（摘要、标点还原、书评）。

**13. `YeungNLP/moss-003-sft-data` (mirror of `fnlp` / `OpenMOSS-Team`) — 670,948 rows in this mirror (official card: 1,074,551 conversations) · CC-BY-4.0 tag / **CC-BY-NC-4.0 per project README** · Tier: Good but half-English / 良好但半英文.**
Sampled offsets 0 / 2,000 / 20,000. Category-labelled multi-turn dialogue (Brainstorming, writing, role play, code, harmless…), built from ~100K **real user inputs** collected during the MOSS-002 beta then answered by `gpt-3.5-turbo` — the prompt distribution is therefore more realistic than self-instruct. | 带类别标签的多轮对话，源自 MOSS-002 内测期收集的约 10 万条**真实用户输入**，再由 `gpt-3.5-turbo` 作答，提示分布比 self-instruct 更贴近真实。
> 如何保障工作中遵循正确的安全准则？ → 1.了解相关安全规定和标准…2.培训和教育…（多轮）
> **Deep-offset:** Stable in quality, but **row 2 of offset 0 is English** ("What best practices should I follow when designing a website?") and English rows keep appearing at every offset — the project states roughly a 50/50 zh/en split. Language-filter before use. | 质量深层稳定，但 offset 0 的第 2 行即为英文，且各偏移处英文行持续出现——项目自述中英各约一半，使用前须按语言过滤。

**14. `Hello-SimpleAI/HC3-Chinese` — 12,853 questions in the `all` config (baike 4,617 · open_qa 3,293 · nlpcc_dbqa 1,709 · psychology 1,099 · medicine 1,074 · finance 689 · law 372) · CC-BY-SA-4.0 · Tier: Good (small, special-purpose) / 良好（小、专用）.**
Sampled offsets 0 and 2,000. Each row pairs **human answers** with **ChatGPT answers** for the same question — designed for detector research, but the `human_answers` side is a clean, small, natively-Chinese resource, and the pairing makes it usable as preference data. | 每行为同一问题配对**人类答案**与 **ChatGPT 答案**。本为检测器研究设计，但 `human_answers` 一侧是干净的小规模原生中文资源，且配对结构可直接用作偏好数据。
> 盗贼天赋盗贼怎么加天赋? → *human*: 搞匕首还加出血（楼上）？天赋看你喜爱了，31 8 12 和 21 8 22 PK都好… / *chatgpt*: 如果你在玩角色扮演游戏（RPG），那么你可能是在问如何在游戏中给你的盗贼角色加天赋…
> **Deep-offset:** Too small to degrade. Note the human answers are forum-register and often terse; the ChatGPT answers are the verbose ones. Licensing is per-source: the `medicine` split's upstream is CC-BY-NC. | 规模过小无退化问题。注意人类答案为论坛语域且常简短，冗长的是 ChatGPT 答案。许可按来源分层：`medicine` 子集上游为 CC-BY-NC。

**15. `Azure99/blossom-v6-sft-stage1` — 149,750 rows · Apache-2.0 · Tier: Good but ~half English / 良好但约半英文.**
Sampled offsets 0 / 2,000 / 20,000. A carefully curated *mixture* with a `metadata.source` field naming each row's origin (`infinity_preference`, `code`, `olcc`, `math`, `wizard`, `magpie`…), i.e. a ready-made recipe rather than a raw dump. Chinese rows are strong. | 精心策划的**混合集**，每行 `metadata.source` 标明来源（`infinity_preference`/`code`/`olcc`/`math`/`wizard`/`magpie` 等），是现成配方而非原始堆料。中文行质量好。
> `source: code` — 您的任务是创建一个使用Modbus RTU协议与远程设备进行通信的程序…（详列常量与参数）
> **Deep-offset:** Consistent, but the very first sampled row is an English LaTeX vector problem and English rows recur throughout — roughly a 1:1 zh:en design. The companion `Azure99/blossom-chat-v3` (5,000 rows) is likewise mixed, with a strong medical zh row at offset 0 (蛛网膜下腔出血后血浆钠下降 / SIADH) and English rows immediately after. | 深层一致，但首行即为英文 LaTeX 向量题，英文行贯穿全集——设计上中英约 1:1。姊妹集 `Azure99/blossom-chat-v3`（5,000 行）同样中英混合。

---

### Tier 3 — Mixed / 参差

**16. `llm-wizard/alpaca-gpt4-data-zh` — 48,818 rows · card tag CC-BY-4.0 but **upstream is CC-BY-NC-4.0** · Tier: Mixed / 参差.**
Sampled offsets 0 and 2,000. GPT-4 answers to *machine-translated* Stanford-Alpaca prompts. The Chinese prose itself is clean and well-formed; the problem is the **task distribution is English**, so you are teaching a Chinese model an American assistant's topic prior. | GPT-4 对**机器翻译**的 Stanford-Alpaca 提示作答。中文行文本身干净规整，问题在于**任务分布是英文的**——等于把美式助手的话题先验教给中文模型。
> 三原色是什么？ → 三原色通常指的是红色、绿色和蓝色（RGB）…此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。
> **Deep-offset:** Uniform. **Important dedup finding: `shibing624/alpaca-zh` is the same data** — 48,818 rows, and its offset-0 rows are byte-identical to `llm-wizard/alpaca-gpt4-data-zh`. Do not mix both. | 分布均匀。**重要去重发现：`shibing624/alpaca-zh` 与之为同一份数据**——同为 48,818 行，offset 0 处逐字节相同，切勿同时混入。

**17. `FreedomIntelligence/Evol-Instruct-Chinese-GPT4` — 70,000 rows · **no license** · Tier: Mixed / 参差.**
Sampled offset 0. Evol-Instruct instructions translated to Chinese, then answered by GPT-4. Answers are long and well-organized — genuinely more substantive than Alpaca-zh. | Evol-Instruct 指令译为中文后由 GPT-4 作答，答案长且组织良好，实质内容明显强于 Alpaca-zh。
> …研究并编写一份关于地震的地质、地震学和社会影响的全面报告… → 一、地震的定义和理解…二、地震发生的原因 1. 构造板块…
> **Deep-offset:** Not probed beyond the head. Visible translation artifact: the response is peppered with "（定量事实1）… （定量事实2）…" — a literal rendering of the English prompt's "(fact 1)" scaffolding that leaked into the Chinese answer. | 未深探。可见翻译残留：答案中反复出现"（定量事实1）（定量事实2）"，是英文提示中 "(fact N)" 支架直译泄漏到中文答案里。

**18. `shibing624/sharegpt_gpt4` — 103,415 rows · CC-BY-4.0 tag, GPT-4-output-derived · Tier: Mixed / 参差.**
Sampled offsets 0 / 2,000 / 20,000 **and** 60,000 / 90,000 / 100,000. **This is the sampling result most likely to surprise:** the dataset is routinely recommended as Chinese SFT data, but **offsets 0 through ~60,000 are entirely English**. Chinese only appears past roughly offset 85,000, and that Chinese is **machine-translated ShareGPT**, not native. | **这是最出人意料的采样结果**：该集常被推荐为中文 SFT 数据，但 **offset 0 至约 60,000 全为英文**；中文要到约 offset 85,000 之后才出现，且是**机器翻译的 ShareGPT**，而非原生中文。
> offset 0: "Summarize the main ideas of Jeff Walker's Product Launch Formula…" (English)
> offset 100,000: 写一封电子邮件给Dave Mitchell，陈述一些关于主题的事实… → 主题：预约会议庆祝您的优秀表现！尊敬的戴夫·米切尔…
> **Deep-offset:** Language flips with offset, not quality. Only the ~38.5K `sharegpt_zh` slice is Chinese; take that slice explicitly rather than the whole repo. | 随偏移变化的是语言而非质量。仅约 3.85 万行的 `sharegpt_zh` 切片为中文，应显式取该切片而非整库。

**19. `CohereLabs/aya_collection_language_split` (`chinese` config) — 58,941 train / 7,397 val / 8,634 test · Apache-2.0 · Tier: Mixed / 参差.**
Sampled offset 0. Templated NLP tasks (NER, classification…) with `dataset_name`, `task_type`, `template_id` and `script: Hans/Hant` fields. Two serious defects: **near-total prompt duplication** (three consecutive rows share an identical ~200-character NER instruction, differing only in the final sentence) and a **Simplified→Traditional conversion bug** — the template says "請識別**下麵**提供的輸入句子", where 下麵 means *noodles*; the intended word is 下面. | 模板化 NLP 任务，带 `dataset_name`/`task_type`/`template_id`/`script` 字段。两处严重缺陷：**提示近乎完全重复**（连续三行共用同一段约 200 字 NER 指令，仅末句不同）；以及**简繁转换错误**——模板写作"請識別**下麵**提供的輸入句子"，"下麵"意为面条，应为"下面"。
> `targets`: "Results": []  ← many rows have empty gold answers / 大量行的标准答案为空
> **Deep-offset:** Not probed; the duplication at offset 0 is severe enough to require aggressive dedup regardless. | 未深探；offset 0 的重复已严重到无论如何都需强力去重。

**20. `CohereLabs/aya_dataset` — 202,362 train rows total, of which **Simplified Chinese 3,038 (1.50%) and Traditional Chinese 1,871 (0.92%)** · Apache-2.0 · Tier: Mixed (tiny zh share) / 参差（中文占比极小）.**
I counted the full `language` column across the whole train split, not a sample. Human-annotated and genuinely high per-example quality, but the corpus is dominated by Plateau Malagasy (7.21%), Sinhala (7.18%), Tamil (6.98%) and Yoruba (5.81%); Chinese is a rounding error. | 我对整个 train 分片的 `language` 列做了全量统计（非抽样）。人工标注、单条质量确实高，但语料以马达加斯加语（7.21%）、僧伽罗语（7.18%）、泰米尔语（6.98%）、约鲁巴语（5.81%）为主，中文只是零头。
> **Deep-offset:** N/A (full-column count). Useful only as a small human-written zh seed (~4.9K rows including Traditional). | 无（已全量统计）。仅可作约 4.9 千行（含繁体）的人写中文种子数据。

**21. `Chinese-Vicuna/instruct_chat_50k.jsonl` — 51,584 rows · Apache-2.0 · Tier: Mixed / 参差.**
Sampled offset 0. Multi-turn, well-formatted (Markdown tables in the answers), but the sampled rows are **Traditional Chinese / Taiwanese register** ("提供一個媒體招待會的流程表…來彰化跑步"), which is a mismatch if you are targeting Simplified-Chinese Mainland usage. | 多轮、格式良好（答案含 Markdown 表格），但采样行为**繁体中文／台湾语域**，若目标是简体中文大陆语用则不匹配。

---

### Tier 4 — Poor for zh general SFT / 不适合中文通用 SFT

**22. `silk-road/Wizard-LM-Chinese-instruct-evol` — 70,000 rows · CC-BY-4.0 · Tier: Poor / 较差.**
Sampled offset 0. The translation pipeline is visibly broken: the very first row's `output_zh` field begins with the literal string **"Translation:"** and then emits **English**, and both `instruction_zh` and `output_zh` are raw HTML fragments wrapped in stray backticks. | 翻译流水线明显损坏：首行 `output_zh` 字段以字面字符串 **"Translation:"** 开头并输出**英文**；`instruction_zh` 与 `output_zh` 均为裹在游离反引号中的原始 HTML 片段。
> `output_zh`: Translation:\n\n<html><body><h3>Healthy Habits:</h3><ul><li>Eat a balanced diet…
> **Deep-offset:** The head alone is disqualifying; use `FreedomIntelligence/Evol-Instruct-Chinese-GPT4` instead for the same source instructions. | 仅开头即已不合格；同源指令请改用 `FreedomIntelligence/Evol-Instruct-Chinese-GPT4`。

**23. `dikw/hh_rlhf_cn` — 344,317 train rows · license tag `llama2` · Tier: Poor / 较差.**
Sampled offsets 0 / 2,000 / 20,000. A machine translation of Anthropic HH-RLHF with severe MT pathologies: degenerate repetition ("大便、大便、大便、大便…"), doubled words ("并不同时同时生活"), and broken punctuation spacing. | Anthropic HH-RLHF 的机器翻译版，MT 病症严重：退化重复（"大便、大便、大便…"）、叠词（"并不同时同时生活"）、标点空格错乱。
> 恐龙制造的什么声音? → 人类和恐龙并不同时同时生活,因此很难说。找到恐龙制造的噪音的最佳地方将是:
> **Deep-offset:** Uniformly bad. | 各偏移一致地差。

**24. `beyond/rlhf-reward-single-round-trans_chinese` — 19,862 train rows · **no license** · Tier: Poor / 较差.**
Sampled offsets 0 and 2,000. Also translated HH-RLHF. Untranslated English leaks into the Chinese ("你对"indoors"是什么理解？"), and in several sampled pairs **both** `chosen` and `rejected` are weak clarifying questions rather than answers, so the preference signal is close to noise. | 同为 HH-RLHF 翻译版。英文残留直接漏入中文（"你对"indoors"是什么理解？"）；多组采样中 `chosen` 与 `rejected` **双方**都只是弱化的澄清提问而非答案，偏好信号接近噪声。
> 今年我想在室内种植水果… → `chosen`: 没问题，你对"indoors"是什么理解？ / `rejected`: 你在考虑哪种水果呢？

**25. `BelleGroup/multiturn_chat_0.8M` — 831,036 rows · GPL-3.0 tag / research-only prose · Tier: Poor *as shipped* / 原样不宜用.**
Sampled offsets 0 / 2,000 / 20,000. The content is fine; the **format is the problem**. One row is *not* one dialogue — it is one assistant reply plus the whole preceding context flattened into the `instruction` string with literal `Human:` / `Assistant:` markers. Loading it naively creates massive context duplication across rows and bakes those role markers into your model. | 内容尚可，**问题在格式**：一行并非一段对话，而是"一条助手回复＋此前全部上下文被压平进 `instruction` 字符串"，并带字面 `Human:` / `Assistant:` 标记。直接加载会造成跨行的上下文大量重复，并把这些角色标记固化进模型。
> `instruction`: Human:你好，你能帮我解答一个问题吗？\nAssistant: 当然，请问有什么问题？\nHuman:我想了解人工智能的未来发展方向… `output`: 人工智能面临的挑战包括数据隐私、安全和道德方面的问题…
> Use `BelleGroup/train_3.5M_CN` (proper `conversations` schema) for multi-turn instead. | 多轮请改用 `BelleGroup/train_3.5M_CN`（正规 `conversations` schema）。

**26. `BelleGroup/school_math_0.25M` — 248,481 rows · GPL-3.0 tag / research-only prose · Tier: Poor (unverified solutions) / 较差（解答未校验）.**
Sampled offsets 0 and 2,000. The card itself warns "题目或解题过程可能包含错误", and the **very first sampled row is wrong**: it converts 10 minutes to 600 seconds, computes 2 km / 600 s = 3.3 m/s, then answers "小明每分钟走3.3米" (the correct answer is 200 m/min). Training on this teaches unit confusion. | 卡片自述"题目或解题过程可能包含错误"，而**首个采样行即错**：把 10 分钟当作 600 秒，算得 2 公里/600 秒 = 3.3 米/秒，答"小明每分钟走3.3米"（正确为 200 米/分）。以此训练会教出单位混淆。
> Superseded by the math slice of `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` (36,568 math rows, R1-generated, `score`-filterable). | 已被 `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` 的数学切片（36,568 行，R1 生成，可按 `score` 过滤）取代。

**27. English-dominant — sampled, measured, and excluded as zh sources / 以英文为主：已采样实测，不作为中文来源.**
These are not bad datasets; they are simply not Chinese ones, and each is routinely miscited as if it were. Where a `language` field exists I counted it in full; where it does not, I used a CJK-density heuristic and say so. | 它们并非质量差，只是不是中文数据，却常被当作中文数据引用。有 `language` 字段者全列统计，无则用中日韩字符密度启发式并已注明。
- `allenai/tulu-3-sft-mixture` — 939,343 rows, ODC-BY. Offsets 0 / 2,000 / 20,000 returned English and Spanish (`ai2-adapt-dev/oasst1_converted`). I then ran a CJK-density scan over **2 of 6 shards = 313,115 rows**: **6,728 rows (2.15%) are Chinese**, and they are almost entirely inherited from one component — `ai2-adapt-dev/tulu_v3.9_wildchat_100k` contributes **6,362 of its 100,000 rows (6.4%)**, with `oasst1_converted` 248/7,131, `flan_v2_converted` 105/89,982 and `personahub_math_v5` 13/106,262. So Tulu 3's Chinese is **WildChat's Chinese, subsampled** — another reason to go to WildChat directly. Excellent English/multilingual mixture; ~2% zh means it is an English anchor, not a zh resource. | 三个偏移采样返回英文与西班牙文。随后对 **6 个分片中的 2 个、共 313,115 行**做中日韩字符密度扫描：**6,728 行（2.15%）为中文**，且几乎全部来自单一成分——`tulu_v3.9_wildchat_100k` 贡献其 10 万行中的 **6,362 行（6.4%）**。即 Tulu 3 的中文就是**被下采样的 WildChat 中文**，不如直接用 WildChat。作为英文锚点优秀，但约 2% 的中文占比说明它不是中文数据来源。
- `Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1` — 1,000,000 rows, 16 shards. I counted the full `language` column over **2 of 16 shards = 125,000 rows**: **EN 124,935 (99.95%)**, Latin 29, Xhosa 12, Indonesian 6, German 5 — and **zero Chinese rows in the sample**. The `pre_query_template` is the plain English Qwen system prompt, i.e. it was **not** steered toward Chinese the way the 200K-Chinese release was. Use `Magpie-Qwen2-Pro-200K-Chinese` instead. | 对 **16 个分片中的 2 个、共 125,000 行**的 `language` 列全量统计：**英语 124,935（99.95%）**，拉丁语 29、科萨语 12、印尼语 6、德语 5，**抽样中中文为零**。`pre_query_template` 是英文 Qwen 系统提示，并未像 200K-Chinese 版那样引导为中文；中文任务请改用 `Magpie-Qwen2-Pro-200K-Chinese`。
- `Skywork/Skywork-Reward-Preference-80K-v0.2` — 77,016 rows. Sampled rows are English (`source: magpie_ultra`). | 77,016 行，采样行为英文。
- `BAAI/Infinity-Preference` — 59,338 train rows, Apache-2.0. Sampled rows are English (PGP/GPG key generation, propositional logic). | 59,338 行，采样行为英文。


---

### Gated / inaccessible — not ranked / 受限或不可访问（未排名）

| Repo | Status observed / 实测状态 | Note / 说明 |
|---|---|---|
| `BAAI/Infinity-Instruct` | `gated: auto`; Parquet branch returned non-Parquet bytes (auth wall) — **could not read a single row** | CC-BY-SA-4.0; 7,449,106 rows in the `7M` config per the paper. **Workaround: its Chinese single-turn slice is redistributed as `Mxode/Chinese-Instruct` config `infinity-instruct`, 386,426 rows, which I did sample.** / 受限；无法读取任何行。可改用 `Mxode/Chinese-Instruct` 的 `infinity-instruct` 子集（386,426 行）。 |
| `lmsys/lmsys-chat-1m` | `gated: auto`; Parquet read failed with an auth wall | 1M real user–LLM conversations with a `language` field; a well-known zh source but requires accepting terms. / 需接受条款。 |
| `BAAI/COIG-PC` | `gated: auto`, 272 Parquet shards listed but blocked | The gate checkbox says "I agree to use this model for **non-commercial use ONLY**". Use `COIG-PC-Lite` (ungated) instead. / 门禁勾选项声明**仅限非商业使用**；改用未设门禁的 `COIG-PC-Lite`。 |
| `BAAI/IndustryInstruction` | `gated: auto`; Parquet read blocked | Apache-2.0 tag, 12 industry verticals. Its zh slice is redistributed as `Mxode/Chinese-Instruct` config `industryinstruction`, **358,582 rows**. / 其中文切片经 `Mxode/Chinese-Instruct` 的 `industryinstruction` 子集转发（358,582 行）。 |
| `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | Not gated, but the HF Parquet-conversion job produced **no configs** — nothing readable through the viewer/Parquet path | **CC-BY-NC-4.0 — non-commercial.** Must be downloaded as raw files. / 未做 Parquet 转换，需下载原始文件；**CC-BY-NC-4.0，非商用**。 |
| `shareAI/ShareGPT-Chinese-English-90k` | Viewer/size job **fails**: `ArrowInvalid: JSON parse error: Column(/category) changed from string to array in row 4` | Apache-2.0 tag on ChatGPT-derived content. Readable only by downloading the raw `.jsonl` files. / 仅能下载原始 jsonl。 |
| `deepctrl/deepctrl-sft-data` | HF API returns **404** to this token | Mirrors on ModelScope / OpenDataLab report 11,381,621 zh + 2,767,403 en rows; treat those digits as second-hand. / 数字来自镜像站，属二手信息。 |
| `m-a-p/neo_sft_phase2` | HF API returns **404 — the repo appears to have been removed** | Survives only as `Mxode/Chinese-Instruct` config `neo_sft_phase2`, **58,549 zh rows**. / 原库已下线，仅存于 `Mxode/Chinese-Instruct` 的 `neo_sft_phase2` 子集（58,549 行）。 |
| `wenge-research/yayi2_sft_data` | **404 — this repo does not exist.** `api/datasets?author=wenge-research` returns only `yayi2_pretrain_data`, `yayi_uie_sft_data`, `yayi_domain_subset`, `TableEval` | Widely cited in dataset lists; it is a propagated citation error. YAYI 2's SFT data was never released. / 该库并不存在，是被反复转引的错误引用；YAYI 2 的 SFT 数据从未发布。 |
| `BAAI/OL-CC` | Gated (401) | Mirror `lorinma/BAAI_OL-CC` exists. **Dual license: CC-BY-NC 4.0 for academic use; tiered commercial fee.** The only fully human-written, crowdsourced Chinese instruction set found (~10k pairs). / 受限，有镜像；**学术 CC-BY-NC 4.0，商用另行收费**；是找到的唯一完全人写、众包的中文指令集（约 1 万条）。 |

---

## 🎯 Recommendation — what I would actually train on / 建议：实际该拿什么训练

**The headline finding of this review: for Chinese, the constraint is not volume, it is *provenance*.** There are well over 10M open zh instruction rows, but the overwhelming majority are one of three things — machine-translated English data, `gpt-3.5-turbo` self-instruct output from 2023, or re-packagings of the previous two. Only four sources in this entire review supply Chinese that a Chinese person actually wrote or that a strong model generated *natively in Chinese*: **WildChat's Chinese slice** (real user prompts), **COIG-CQIA** (human answers, human-verified), **Magpie-Qwen2-Pro-200K-Chinese** (natively self-synthesized by Qwen2-72B), and **Chinese-DeepSeek-R1-Distill-110k** (R1 reasoning on Chinese prompts). Everything else should be treated as filler or as breadth, not as the thing that teaches your model how Chinese assistants talk.
/ **本综述的核心结论：中文的瓶颈不是数据量，而是数据来源。** 公开中文指令数据远超一千万条，但绝大多数不外乎三类——英文数据的机器翻译、2023 年 `gpt-3.5-turbo` 的 self-instruct 产物，或前两者的再打包。整篇综述中只有四个来源提供了**中国人真正写的**或**强模型以中文原生生成的**中文：**WildChat 的中文切片**（真实用户提问）、**COIG-CQIA**（人写答案且经人工校验）、**Magpie-Qwen2-Pro-200K-Chinese**（Qwen2-72B 原生自合成）、**Chinese-DeepSeek-R1-Distill-110k**（R1 在中文提示上的推理）。其余应视为填充或广度补充，而非教会模型"中文助手怎么说话"的主体。

**Given that this project does continued pretraining of Qwen-class models on zh/hi/ar and now wants an SFT stage, I would build a ~600K–800K-example zh-centric mixture as follows** / **鉴于本项目是在 zh/hi/ar 上对 Qwen 系模型做持续预训练、现需加入指令微调阶段，我会按下表构建约 60–80 万条的中文为主混合：**

| # | Component / 成分 | Take / 取用量 | Why / 理由 | License risk / 许可风险 |
|---|---|---|---|---|
| 1 | `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese`, filtered `instruct_reward > 0` and `input_quality ∈ {good, excellent}`, deduped on `min_neighbor_distance` | ~120K | Natively-Chinese general chat backbone, **generated by a Qwen-family teacher** — the closest possible match to a Qwen-class student's own distribution. Ships its own reward/quality/dedup metadata. | ⚠️ No license tag; Qwen2-72B-Instruct outputs (Qwen license) |
| 2 | `allenai/WildChat-1M`, `language == "Chinese"`, `toxic == false`, `redacted == false`, length- and turn-filtered | ~150K | **Real Chinese user prompts.** Nothing else in the open ecosystem fixes the prompt-distribution gap. Multi-turn comes free (`turn` field). | ⚠️ ODC-BY, but responses are OpenAI outputs |
| 3 | `Congliu/Chinese-DeepSeek-R1-Distill-data-110k`, `score >= 9` | ~70K | zh reasoning + long-form CoT; strictly better than BELLE `school_math` (whose first sampled row is arithmetically wrong). Drop or keep `reasoning_content` depending on whether you want a thinking model. | ✅ Apache-2.0 |
| 4 | `m-a-p/COIG-CQIA`, `human_verified == true`, **drop the `xhs` config**, upsample ×2–3 | ~80K (×2) | The only human-written Chinese answers at usable scale. Upsample because it is small and it is the only thing anchoring native Chinese register. Its `ruozhiba` subset is directly relevant to figurative/idiomatic understanding. | ⚠️ **No license declared** — the biggest unresolved risk in the mix |
| 5 | `Mxode/Chinese-Instruct`, configs `stem_zh` + `firefly` + `infinity-instruct` + `neo_sft_phase2`, subsampled | ~200K | Breadth: STEM explanation, 23 Chinese NLP task types, and the zh slice of Infinity-Instruct — all with a declared license, which the originals lack. **Skip the `magpie` config** (translationese, sister/elder-sister slip at depth). | ✅ CC-BY-SA-4.0 (share-alike is copyleft — check it is compatible with your release plan) |
| 6 | `YeungNLP/firefly-train-1.1M`, subsample the 对联 / 古诗 / 文言文 / 金庸 task types only | ~20K | **Chinese-cultural task types that exist nowhere else**, and it is not GPT-distilled. Directly on-topic for a figurative-language project. | ⚠️ **No license at all** |
| 7 | English/multilingual anchor: `allenai/tulu-3-sft-mixture`, subsampled | ~80K (10%) | Prevents the catastrophic forgetting of English and general instruction-following that a zh-only SFT stage causes on a Qwen base. Tulu 3 is the best-documented open post-training mixture. ⚠️ Dedup against component 2 — its ~2.15% Chinese comes from a 100K subsample of WildChat. | ✅ ODC-BY |

**Explicitly do NOT include** / **明确不要加入**: `silk-road/Wizard-LM-Chinese-instruct-evol` (broken translation, `output_zh` emits English), `dikw/hh_rlhf_cn` and `beyond/rlhf-reward-single-round-trans_chinese` (degenerate MT), `BelleGroup/school_math_0.25M` (wrong solutions), `BelleGroup/multiturn_chat_0.8M` **in its shipped flattened format**, `shibing624/alpaca-zh` **if you already have** `llm-wizard/alpaca-gpt4-data-zh` (identical data), and `Magpie-Qwen2.5-Pro-1M-v0.1` (99.95% English in my count).

**Preference / DPO stage** / **偏好与 DPO 阶段**: the open Chinese preference landscape is genuinely weak. `BAAI/Infinity-Preference` (59,338) and `Skywork/Skywork-Reward-Preference-80K-v0.2` (77,016) are **English** in every row I sampled; `dikw/hh_rlhf_cn` and `beyond/rlhf-...-trans_chinese` are broken machine translations. The two usable options are **`COIG-P`** (1,009k zh preference pairs across Chat/Code/Math/Logic/Novel/Role, built with 15 LLMs, arXiv:2504.05535) and **`Hello-SimpleAI/HC3-Chinese`** repurposed as human-vs-ChatGPT preference pairs (12,853 questions). If neither fits, generating on-policy preference pairs over your own SFT prompts is likely to beat anything currently public in Chinese. | 开源中文偏好数据确实薄弱：`Infinity-Preference` 与 `Skywork-Reward-Preference-80K-v0.2` 在我采样的每一行都是英文；两个 HH 中译版本质量破损。可用者仅 **COIG-P**（约 100.9 万条中文偏好对）与把 **HC3-Chinese** 改用为"人类 vs ChatGPT"偏好对。若都不合适，在自有 SFT 提示上做 on-policy 偏好采样，很可能优于目前任何公开中文偏好集。

**Totals and overlaps** / **总量与重叠**: the seven components above sum to roughly **720K examples**, of which ~89% is Chinese. Two overlaps must be deduplicated before training: component 7's Chinese rows are drawn from the same WildChat pool as component 2, and component 5's `coig-cqia` subset (10,306 rows, not selected above) overlaps component 4. Deduplicate on normalized instruction text. | 上述七项合计约 **72 万条**，其中约 89% 为中文。训练前须去重两处重叠：成分 7 的中文行与成分 2 同源于 WildChat；成分 5 的 `coig-cqia` 子集与成分 4 重叠。建议按归一化指令文本去重。

**One structural warning** / **一个结构性提醒**: **do not sum the BELLE datasets.** `train_0.5M_CN` (519,255), `train_1M_CN` (917,424), `train_2M_CN` (2,000,000) and `train_3.5M_CN` (3,606,402) are cumulative releases from the same pipeline, not disjoint corpora; adding them yields ~8.5M rows with heavy duplication. Pick one — `train_3.5M_CN`, which is the only one in proper `conversations` format. | **不要把 BELLE 各集相加**：它们是同一流水线的累进发布而非互斥语料，相加得到约 850 万行且高度重复。只取其一——`train_3.5M_CN`（唯一采用规范 `conversations` 格式者）。

---

> **Scope / 范围:** This review covers **general-purpose** open Chinese instruction-tuning / SFT / post-training data: general chat and assistant data, reasoning / math / code SFT, multi-turn dialogue, human-written vs. synthetic vs. distilled vs. translated construction, multilingual mixtures with a substantial Chinese portion, and (as a secondary section) preference / DPO / RLHF data. It is **not** restricted to culture or idioms. It also covers what the major Chinese model technical reports say about their SFT mixtures — and, importantly, whether they released them (almost none did).
> 本综述覆盖**通用**开源中文指令微调 / SFT / 后训练数据：通用对话与助手数据、推理/数学/代码 SFT、多轮对话、人写 vs 合成 vs 蒸馏 vs 翻译的构建方式、含大量中文的多语混合，以及（作为次要章节）偏好/DPO/RLHF 数据。**不限于**文化或习语主题。同时覆盖各主要中文大模型技术报告对其 SFT 混合的描述——以及一个关键问题：它们是否公开了这些数据（几乎都没有）。

> **Method / 方法说明:** Three streams were run in parallel. (1) **Hands-on sampling**: real rows read from HuggingFace Parquet shards via HTTP range reads (see the ranking's method note); full-column language counts were computed where a `language` field exists. (2) **Source verification**: dedicated sub-agents fetched arXiv abstracts/full texts, HF dataset cards and GitHub READMEs, and were instructed to mark anything they could not confirm as *unverified*. (3) **Metadata verification**: `huggingface.co/api/datasets/*` and `datasets-server.huggingface.co/size` were queried directly for gating status, license tags and true row counts. **All numbers in this document are real** — card claims and observed values are shown side by side where they disagree, and items that could not be verified are flagged rather than filled in.
> 三条工作流并行：(1) **实测抽样**——通过 HTTP Range 直读 HF Parquet 分片获取真实数据行（方法详见排名节），凡有 `language` 字段者做全列语言统计；(2) **来源核实**——由专门子代理抓取 arXiv 摘要/全文、HF 数据卡与 GitHub README，凡无法确认者一律标注 *unverified*；(3) **元数据核实**——直接查询 `huggingface.co/api/datasets/*` 与 `datasets-server` 的 `/size`，获取门禁状态、许可标签与真实行数。**文中所有数字均为实测或原始出处**；卡片声称值与实测值不符时并列展示，无法核实者予以标注而非填补。

---

## Taxonomy / 分类导览

- **(A) Human-written / human-verified Chinese / 人写或人工校验的中文:** COIG-CQIA, BAAI/OL-CC, Zhihu-KOL, HC3-Chinese (human side), Aya Dataset (zh slice), COIG (exam & human-value subsets).
- **(B) Real user prompts, model answers / 真实用户提问＋模型作答:** WildChat-1M (zh 23.27%), LMSYS-Chat-1M (gated), MOSS-003 (built on ~100K beta-user inputs), ShareGPT-derived sets.
- **(C) Self-instruct / distilled from a teacher LLM / self-instruct 与教师模型蒸馏:** BELLE family (davinci-003 / gpt-3.5-turbo), MOSS-002/003, Alpaca-zh, alpaca-gpt4-data-zh (GPT-4), Evol-Instruct-Chinese-GPT4, TigerBot alpaca-zh, Blossom.
- **(D) Prompt-free self-synthesis / 无提示自合成:** Magpie family (Qwen2-72B-Instruct, Qwen2.5-72B-Instruct).
- **(E) Instruction back-translation / 指令回译:** COIG-Kun (Yi-based, over Wudao / WanJuan / SkyPile).
- **(F) Template-over-existing-NLP-datasets / 既有 NLP 数据套模板:** Firefly (23 zh tasks), COIG-PC / COIG-PC-Lite (3,339 tasks), Aya Collection (zh config), pCLUE.
- **(G) Reasoning distillation (R1 era) / 推理蒸馏（R1 时代）:** Chinese-DeepSeek-R1-Distill-110k, Mxode Chinese-Reasoning-Distil, AM-DeepSeek-R1-Distilled-1.4M (CC-BY-NC).
- **(H) Curated mega-mixtures / 策展巨型混合:** Infinity-Instruct (7.4M, gated), Mxode/Chinese-Instruct (4.85M), deepctrl-sft-data (~12M, gated), Tulu 3 (English-centric).
- **(I) Multilingual mixtures with a zh portion / 含中文的多语混合:** Aya Dataset & Aya Collection, Tulu 3, Magpie multilingual releases, WildChat, LMSYS-Chat-1M.
- **(J) Preference / DPO / RLHF (zh) / 中文偏好数据:** COIG-P, CValues, Infinity-Preference (en), Skywork-Reward-Preference (en), hh_rlhf_cn (MT), Chinese-dpo-pairs, zhihu_rlhf_3k.
- **(K) Model technical reports describing but **not** releasing SFT data / 描述但**未公开** SFT 数据的模型报告:** Qwen 1/2/2.5/3, Yi, InternLM2, DeepSeek LLM/V3/R1, Baichuan 2, GLM-4, MiniCPM, Skywork, TeleChat, Hunyuan-Large.

---

## 1. Human-Written & Human-Verified Chinese Instruction Data / 一、人写与人工校验的中文指令数据

### COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning (Bai et al., 2024)
- **Venue / Link:** **Findings of NAACL 2025**, pp. 8205–8220 — https://aclanthology.org/2025.findings-naacl.457/ · arXiv:2403.18058, https://arxiv.org/abs/2403.18058
- **Data / 数据:** https://huggingface.co/datasets/m-a-p/COIG-CQIA (canonical, ungated). A `BAAI/COIG-CQIA` mirror exists but is **gated (401)**.
- **Size / 规模:** **44,694 rows** across 13 viewer configs (verified via `datasets-server`); the paper's own table totals **48,384** — flag the mismatch. Per config: finance 11,288 · wiki 10,603 · zhihu 5,631 · exam 4,856 · douban 3,086 · coig_pc 3,000 · xhs 1,508 · wikihow 1,485 · chinese_traditional 1,111 · human_value 1,007 · segmentfault 458 · logi_qa 421 · ruozhiba 240. No token count published.
- **Construction / 构建方式:** Real Chinese web content (Zhihu, Douban, Xiaohongshu, SegmentFault, MBA智库, wikiHow-zh, Baidu/Chinese Wikipedia, exams, 100PoisonMpts) → rule-based filtering → template construction → **manual verification**, recorded per-row in `answer_from` (`human`/`llm`) and `human_verified` (bool). **Not GPT-distilled.**
- **Domains / 领域:** Social media & forums (13,935), finance/business (11,289), medicine (8,537), encyclopedia (4,571), NLP tasks (3,000), exams & logic (2,897), law (2,645), human values (1,007), traditional culture (503).
- **License / 许可:** **None declared.** The card says "[More Information Needed]" and the HF API returns no license tag. Per-row `copyright` field is mostly "暂无版权及作者信息". **Treat as unlicensed, not permissive.**
- **Motivation & quality / 动机与质量:** The paper's thesis is exactly the gap this review confirms — "existing datasets, generally distilled from English-centric LLMs, are not well-aligned with Chinese users' interaction patterns". My sampling agrees: this is the most natural-sounding Chinese in the review, and its `ruozhiba` subset (240 rows of Chinese internet logic/pun questions, tagged `minor: 逻辑问答, 隐喻理解`) is uniquely on-topic for figurative-language work. The `xhs` config is emoji marketing spam and should be dropped. | 论文主张恰好印证本综述的结论——现有数据多由英文中心模型蒸馏而来，与中文用户交互模式不匹配。实测显示这是全篇最自然的中文；其 `ruozhiba` 子集（240 条，标注含"隐喻理解"）对比喻性语言研究尤为契合；`xhs` 子集为表情营销文案，应剔除。

### OL-CC — 开源中文指令语料 (BAAI, 2023)
- **Venue / Link:** No paper. Announcement: https://hub.baai.ac.cn/view/26120
- **Data / 数据:** `BAAI/OL-CC` on HF is **gated (401)**; community mirror https://huggingface.co/datasets/lorinma/BAAI_OL-CC
- **Size / 规模:** ~**10k+ instruction–answer pairs plus 1.6k+ human-written instructions**; an earlier phase reported 270 volunteers producing 4.7k pairs + 5.3k instructions over two months.
- **Construction / 构建方式:** **Crowdsourced, fully human-written** — the only such Chinese instruction set found in this review. Task types: QA, writing, extraction, editing, classification, brainstorming, chat, logic/math.
- **License / 许可:** ⚠️ **Dual — CC-BY-NC 4.0 for academic use; a tiered commercial fee (free under 1M users).** Flag as non-commercial by default.
- **Quality / 质量:** Not sampled (gated). Its value is categorical rather than quantitative: at ~10k rows it cannot be a backbone, but it is the only source of Chinese instructions with no model in the loop at all. It shows up as a labelled `source: olcc` component inside `Azure99/blossom-v6-sft-stage1`, which I did sample. | 未能采样（受限）。价值在于性质而非规模：约 1 万条不足以作主干，但它是唯一全流程无模型参与的中文指令来源；在我实际采样的 `blossom-v6-sft-stage1` 中以 `source: olcc` 的标注成分出现。

### Zhihu-KOL (wangrui6, 2023)
- **Venue / Link:** No paper.
- **Data / 数据:** https://huggingface.co/datasets/wangrui6/Zhihu-KOL
- **Size / 规模:** **1,006,218 rows** (5 Parquet shards, ~2.3 GB in memory). Fields: `INSTRUCTION`, `RESPONSE`, `SOURCE`, `METADATA` (with `question_id`, `answer_id`, `url`, `upvotes`, `answer_creation_time`).
- **Construction / 构建方式:** **Scraped human-written Zhihu answers** from key-opinion-leader accounts. No model in the loop. Originally collected for Open Assistant.
- **License / 许可:** ⚠️ **No license declared at all.** This is scraped copyrighted user content from a commercial platform — the highest legal-risk item in this review.
- **Quality / 质量:** Authentic Chinese voice at a scale nothing else matches, but **quality falls off with depth** (see ranking #8): high-upvote answers at offset 0 are essays; at offset 20,000 I hit a one-line hedge. Threshold on `METADATA.upvotes` and response length. | 中文语气之真实无出其右，但**深层质量下滑**（见排名第 8 条）；须按 `upvotes` 与长度过滤。

### HC3 — Human ChatGPT Comparison Corpus, Chinese (Guo et al., 2023)
- **Venue / Link:** arXiv:2301.07597, https://arxiv.org/abs/2301.07597 (arXiv preprint; no venue found)
- **Data / 数据:** https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese · https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- **Size / 规模:** **12,853 questions** in the `all` config — baike 4,617 · open_qa 3,293 · nlpcc_dbqa 1,709 · psychology 1,099 · medicine 1,074 · finance 689 · law 372. Each row holds a **list** of `human_answers` and a **list** of `chatgpt_answers`, so answer counts exceed the row count.
- **Construction / 构建方式:** Hybrid — questions and `human_answers` are existing human-expert data from public Chinese QA corpora; `chatgpt_answers` were generated by ChatGPT. **Designed as a detection corpus, not an SFT set**, though it is widely reused as one (e.g. TigerBot's `HC3-zh-12k`).
- **License / 许可:** CC-BY-SA-4.0, with per-source escalation. ⚠️ The `medicine` split's upstream (Chinese Medical Dialogue) is **CC-BY-NC 4.0**; WebTextQA/BaikeQA are MIT; NLPCC-DBQA and LegalQA are unknown. Share-alike is itself copyleft for downstream mixes.
- **Quality / 质量:** Small but clean, and the pairing is unusually useful: it gives you a ready-made "human vs. model" contrast in Chinese, which is scarce. Note the register asymmetry my sampling showed — human answers are terse and forum-flavoured, ChatGPT answers are the long verbose ones. | 规模小但干净，且成对结构给出稀缺的中文"人类 vs 模型"对照。注意实测中的语域不对称：人类答案简短且带论坛味，冗长的是 ChatGPT 答案。

---

## 2. Real User Prompts with Model Answers / 二、真实用户提问＋模型作答

### WildChat: 1M ChatGPT Interaction Logs in the Wild (Zhao et al., 2024)
- **Venue / Link:** **ICLR 2024**; arXiv:2405.01470, https://arxiv.org/abs/2405.01470
- **Data / 数据:** https://huggingface.co/datasets/allenai/WildChat-1M
- **Size / 规模:** **837,989 conversations** (`datasets-server` `/size`), 14 Parquet shards. **Measured language mix over 3 shards / 179,571 rows: English 47.70%, Chinese 23.27% (41,787 rows), Russian 11.93%, French 2.40%, Spanish 2.32%** → roughly **~195K Chinese conversations** overall.
- **Construction / 构建方式:** Real users were given free access to `gpt-3.5-turbo` and `gpt-4` in exchange for their conversation logs. **The prompts are genuine; the responses are OpenAI model outputs.** Per-turn metadata includes `country`, `state`, `hashed_ip`, `header.accept-language`, `language`, `turn`, `toxic`, `redacted`, plus full `openai_moderation` and `detoxify_moderation` score vectors.
- **License / 许可:** **ODC-BY** (dataset). ⚠️ Responses are OpenAI outputs → OpenAI ToU competing-model restrictions apply independently of the ODC-BY tag.
- **Quality / 质量:** The single most important finding of my sampling. Chinese rows pulled from shard 8/14 are ordinary, realistic Chinese requests with province-level provenance ("除了猫狗，在家值得养的宠物有哪些，可以互动的", `country: China / state: Zhejiang`; "大中小学思政课优质课程资源共建共享的现实策略…请写对策不少于500字", `country: Hong Kong`) — exactly the prompt distribution that translated Alpaca data cannot supply. ⚠️ It is a **log, not a curated set**: expect homework-mill requests and NSFW roleplay; filter on `toxic`, `redacted`, `turn` and length. | 这是实测中最重要的发现。第 8 个分片取出的中文行是带省级溯源的普通、真实中文请求，正是翻译版 Alpaca 无法提供的提示分布。⚠️ 它是**日志而非策展集**：含大量作业代写与 NSFW 角色扮演，须按毒性/脱敏/轮次/长度过滤。

### LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset (Zheng et al., 2024)
- **Venue / Link:** ICLR 2024; arXiv:2309.11998, https://arxiv.org/abs/2309.11998
- **Data / 数据:** https://huggingface.co/datasets/lmsys/lmsys-chat-1m — ⚠️ **`gated: auto`**; my token could not read the Parquet branch.
- **Size / 规模:** 1M conversations with 25 LLMs from ~210K unique IPs on Chatbot Arena; a `language` field is present. Chinese share **not verified here** (gated).
- **Construction / 构建方式:** Real Chatbot Arena traffic; responses come from whichever of 25 models served the turn (open and proprietary).
- **License / 许可:** Custom LMSYS-Chat-1M dataset license accepted at the gate; **not verified here**.
- **Quality / 质量:** **Not sampled — gated.** Listed for completeness; WildChat is the ungated equivalent and was sampled instead. | **未采样——受限**。列出以求完整；WildChat 是可直接读取的等价物，已实测。

### MOSS: An Open Conversational Large Language Model (Sun et al., 2024)
- **Venue / Link:** **Machine Intelligence Research 21(5): 888–905, 2024**, DOI 10.1007/s11633-024-1502-8
- **Data / 数据:** https://huggingface.co/datasets/fnlp/moss-003-sft-data (now redirects to `OpenMOSS-Team/moss-003-sft-data`) · mirror https://huggingface.co/datasets/YeungNLP/moss-003-sft-data (**670,948 rows**, the version I sampled) · https://github.com/OpenMOSS/MOSS
- **Size / 规模:** **moss-002-sft-data: 1,161,137 conversations** (en_helpfulness 419,049 · en_honesty 112,580 · en_harmlessness 38,873 · zh_helpfulness 447,750 · zh_honesty 142,885). **moss-003-sft-data: 1,074,551 conversations** (writing 341,087 · role playing 246,375 · code 198,079 · brainstorming 99,162 · complex instruction 95,574 · harmless 74,573 · other 19,701). Plugin data ≈300K–350K.
- **Construction / 构建方式:** moss-002 = Self-Instruct expansion of human seed prompts, **distilled from `text-davinci-003`**; English harmlessness prompts from Anthropic red-teaming. moss-003 = built on **~100K real user inputs collected during the MOSS-002 beta**, answered by **`gpt-3.5-turbo`** — a notably more realistic prompt distribution than pure self-instruct. Honesty data was withheld from moss-003 for privacy reasons.
- **License / 许可:** ⚠️ **Conflict.** HF tags both repos `cc-by-4.0`; the **project README states CC BY-NC 4.0** for all data releases (code Apache-2.0, model AGPL-3.0). Cite the README. Also davinci-003/gpt-3.5-turbo-derived → OpenAI ToU.
- **Quality / 质量:** Fluent, category-labelled, genuinely multi-turn, stable to offset 20,000 in my sampling. ⚠️ **Only about half is Chinese** — moss-002 is 50.7% zh by row count, moss-003 is ~50/50 per the README, and row 2 of offset 0 in my sample is English. Language-filter before use. | 流畅、带类别标签、真多轮，实测至 offset 20,000 稳定。⚠️ **仅约一半为中文**，使用前须按语言过滤。

---

## 3. Self-Instruct & Teacher-Distilled Chinese Data / 三、Self-Instruct 与教师模型蒸馏的中文数据

### Exploring the Impact of Instruction Data Scaling on Large Language Models — BELLE (Ji et al., 2023)
- **Venue / Link:** arXiv:2303.14742, https://arxiv.org/abs/2303.14742 (arXiv preprint only). Related: arXiv:2304.07854 (BELLE-0.5M-CLEAN). Code: https://github.com/LianjiaTech/BELLE (Apache-2.0)
- **Data & Size / 数据与规模 (verified row counts, not card claims):**
  | Repo | Card claim | **Actual rows** | Schema |
  |---|---|---|---|
  | `BelleGroup/train_0.5M_CN` | ~50万 | **519,255** | instruction/input/output |
  | `BelleGroup/train_1M_CN` | ~100万 | **917,424** | instruction/input/output |
  | `BelleGroup/train_2M_CN` | ~200万 | **2,000,000** | instruction/input/output |
  | `BelleGroup/train_3.5M_CN` | ~350万 | **3,606,402** | id/conversations |
  | `BelleGroup/multiturn_chat_0.8M` | ~80万 | **831,036** | flattened turns |
  | `BelleGroup/school_math_0.25M` | ~25万 | **248,481** | instruction/input/output |
  | `BelleGroup/generated_chat_0.4M` | ~40万 | **396,004** | instruction/input/output |
- **Construction / 构建方式:** Stanford Alpaca's 175 seeds translated to Chinese and **culturally localized** ("modify some of the data that heavily involve Western culture … to be more in line with Chinese cultural and background knowledge"), then used as in-context examples to have OpenAI models generate more. The released generator defaults to `text-davinci-003` with `gpt-3.5-turbo` as an option — **which teacher produced which shard is unverified**. `train_3.5M_CN` has a verified 13-category domain breakdown via `BELLE-2/train_3.5M_CN_With_Category`: generation 34.61%, role playing 15.29%, open qa 13.87%, brainstorming 6.35%, classification 6.01%, code 5.96%, math 4.65%, close qa 3.67%, summarization 3.27%, translation 3.01%, harmless 1.64%, rewrite 1.39%, extract 0.28%.
- **License / 许可:** ⚠️ **Contradiction.** Every card's YAML says `license: gpl-3.0`, while every card's prose says research-only: "仅允许将此数据集及使用此数据集生成的衍生物用于研究目的，不得用于商业". GPL-3.0 permits commercial use, so tag and prose conflict. Plus OpenAI-output-derived. **Treat the whole family as research-only.**
- **Quality / 质量:** `train_3.5M_CN` is the best of the family and a legitimately useful multi-turn zh source (ranking #6), but **instruction-following failures persist at depth** — at offset 20,000 a "总结他们在聊什么" instruction is answered by continuing the dialogue. `school_math_0.25M` is worse: the card admits solutions may be wrong and my **first sampled row is arithmetically wrong** (2 km / 10 min answered as "3.3米/分钟"). `multiturn_chat_0.8M` is unusable as shipped because one row is one reply plus flattened context. | `train_3.5M_CN` 为家族最佳、确实可用（排名第 6），但**深层仍存在指令遵循失败**；`school_math_0.25M` 首个采样行即算错；`multiturn_chat_0.8M` 因"一行＝一条回复＋压平上下文"的格式原样不可用。
- **⚠️ Do not sum these.** They are cumulative releases from one pipeline, not disjoint corpora; no dedup/overlap statistics are published. | **切勿相加**：它们是同一流水线的累进发布，官方未公布去重/重叠统计。

### Instruction Tuning with GPT-4 — alpaca-gpt4-data-zh (Peng et al., 2023)
- **Venue / Link:** arXiv:2304.03277, https://arxiv.org/abs/2304.03277 · https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
- **Data / 数据:** https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh (canonical; `c-s-ale/alpaca-gpt4-data-zh` redirects here). **`shibing624/alpaca-zh` is the same data.**
- **Size / 规模:** **48,818 rows** (both repos), below the paper's advertised "52K".
- **Construction / 构建方式:** Two-stage distillation — Stanford Alpaca prompts **machine-translated into Chinese by ChatGPT**, then **answers generated by GPT-4**.
- **License / 许可:** ⚠️ The HF card is tagged `cc-by-4.0`, but the upstream GitHub release states **CC BY-NC 4.0, research use only**. Cite the upstream. Also GPT-4-derived → OpenAI ToU.
- **Quality / 质量:** The Chinese prose is clean, well-structured and factually careful (the RGB/RYB answer even distinguishes additive from subtractive colour mixing). The problem is not fluency but **provenance of the task distribution**: the prompts are translated English, so the topic prior is American. Fine as a small clean seed; wrong as a backbone. | 中文行文干净、结构清晰、事实谨慎；问题不在流畅度而在**任务分布来源**——提示为英译中文，话题先验是美式的。可作干净小种子，不宜作主干。

### Chinese-LLaMA-Alpaca instruction data — alpaca_data_zh_51k (Cui et al., 2023)
- **Venue / Link:** arXiv:2304.08177, https://arxiv.org/abs/2304.08177 · https://github.com/ymcui/Chinese-LLaMA-Alpaca
- **Data / 数据:** `data/alpaca_data_zh_51k.json` in the repo; HF re-host https://huggingface.co/datasets/hfl/alpaca_zh_51k
- **Size / 规模:** **51,179 rows.** Later Chinese-LLaMA-Alpaca versions scale instruction tuning to ~2M (basic) and ~4.3M (Plus), drawing on pCLUE, translation data, Stanford Alpaca, self-instruct crawls, STEM data and OASST1.
- **Construction / 构建方式:** The repo's own `data/README.md` says the 51k set was **crawled from ChatGPT (`gpt-3.5-turbo`)** — i.e. distilled, **not** a translation of Stanford Alpaca 52K.
- **License / 许可:** ⚠️ The source repo states "本项目相关资源仅供学术研究之用，严禁用于商业用途" (academic research only). The `hfl/alpaca_zh_51k` re-host is tagged `apache-2.0`, which **contradicts** the source terms — cite the GitHub restriction.
- **Quality / 质量:** Not separately sampled (superseded in practice by `alpaca-gpt4-data-zh`, which uses a stronger teacher on the same prompt family). | 未单独采样（实践中已被采用更强教师、同一提示族的 `alpaca-gpt4-data-zh` 取代）。

### Evol-Instruct Chinese variants / Evol-Instruct 中文变体
- **`FreedomIntelligence/Evol-Instruct-Chinese-GPT4`** — https://huggingface.co/datasets/FreedomIntelligence/Evol-Instruct-Chinese-GPT4 · **70,000 rows** · construction: WizardLM Evol-Instruct-70K questions translated to Chinese, then **GPT-4 generates Chinese responses** · **license: none declared** · Related: LLMZoo / AceGPT (arXiv:2309.12053, arXiv:2304.10453). **Quality:** long, well-organized answers — the best of the Evol-Instruct-zh family — but with visible translation scaffolding leaking into the output ("（定量事实1）… （定量事实2）"). | 答案长且组织良好，为该族最佳；但英文提示的支架直译泄漏进中文答案。
- **`silk-road/Wizard-LM-Chinese-instruct-evol`** — https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol · **70,000 rows** · CC-BY-4.0 · part of the Luotuo (骆驼) project. **Quality: Poor.** The card itself notes instruction-injection failures during translation, and my first sampled row confirms it: `output_zh` literally begins "Translation:" and then emits English, wrapped in raw HTML. **Do not use.** | **质量差**：卡片自述翻译期间存在指令注入失败，实测首行 `output_zh` 以 "Translation:" 开头并输出英文、内含原始 HTML。**不建议使用。**

### TigerBot SFT Chinese data (Chen et al., 2023)
- **Venue / Link:** arXiv:2312.08688, https://arxiv.org/abs/2312.08688 · https://github.com/TigerResearch/TigerBot
- **Data / 数据:** https://huggingface.co/datasets/TigerResearch/sft_zh
- **Size / 规模:** **530,705 rows** — `tigerbot-alpaca-zh-0.5m` 500,000 (≈94%) · `tigerbot-HC3-zh-12k` 12,000 · `tigerbot-zhihu-zh-10k` 10,000 · `tigerbot-superclue-c3-zh-5k` 5,000 · `tigerbot-wiki-qa-zh-1k` 1,000 · `tigerbot-book-qa-1k` 1,000 · `tigerbot-riddle-qa-1k` 1,000. Parallel English side: `TigerResearch/sft_en`.
- **Construction / 构建方式:** ~94% is TigerBot's own Alpaca-style Chinese QA generation; **the generating model is never named — unverified**, though Alpaca-style generation implies an OpenAI teacher. The remaining ~6% repackages public data (HC3-zh, Zhihu, SuperCLUE-C3, wiki/book/riddle QA).
- **License / 许可:** `apache-2.0` on both `sft_zh` and `tigerbot-alpaca-zh-0.5m`. ⚠️ Two flags: the HC3-zh component inherits CC-BY-SA-4.0 (and CC-BY-NC upstream for its medical source), which Apache-2.0 does not cover; and the 500K core is probably OpenAI-derived.
- **Quality / 质量:** Clean and consistent, and — unusually — **it improves with depth**: offset 0 is dominated by Chinese-instruction-over-English-content translation drills, while offsets 2,000 and 20,000 are fully Chinese and more assistant-like. | 干净一致，且**深层反而更好**：offset 0 多为"中文指令＋英文正文"的翻译练习，offset 2,000 与 20,000 为纯中文且更接近助手行为。

### Blossom (Azure99)
- **Venue / Link:** No paper. https://huggingface.co/Azure99
- **Data / 数据:** `Azure99/blossom-v6-sft-stage1` (**149,750 rows**, Apache-2.0), `Azure99/blossom-chat-v3` (**5,000 rows**, Apache-2.0)
- **Construction / 构建方式:** A curated *recipe* rather than a raw dump: every row carries `metadata.source` naming its origin (`infinity_preference`, `code`, `olcc`, `math`, `wizard`, `magpie`, ShareGPT, WildChat, StackOverflow, Flan, Ruozhiba…). The v5 training protocol is stage 1 = 40K Wizard + 40K Orca + 10K Math (1 epoch), stage 2 = 10K Blossom chat multi-turn + 10% stage-1 resample (3 epochs).
- **License / 许可:** Apache-2.0 tag on the datasets sampled; components carry their own upstream terms.
- **Quality / 质量:** Chinese rows are strong and the provenance labelling is a real asset for building a controlled mixture. ⚠️ Roughly **1:1 Chinese:English by design** — the very first sampled row is an English LaTeX vector problem. | 中文行质量好，逐行来源标注对构建可控混合极有价值。⚠️ 设计上中英约 1:1。

---

## 4. Prompt-Free Self-Synthesis — the Magpie family / 四、无提示自合成——Magpie 系列

### Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing (Xu et al., 2024)
- **Venue / Link:** **ICLR 2025**; arXiv:2406.08464, https://arxiv.org/abs/2406.08464 · https://github.com/magpie-align/magpie · https://magpie-align.github.io/
- **Method / 方法:** Feed an aligned LLM only its own pre-query chat template (e.g. `<|im_start|>user\n`) and let it auto-complete a user instruction, then let it answer. No seed prompts, no human labour, no translation — so the language of the output is whatever the teacher natively produces.
- **Chinese releases and sizes / 中文相关发布与规模:**
  - **`Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese`** — **200,000 rows**, generated by `Qwen/Qwen2-72B-Instruct`, card declares `language: zh`, per-row `language: ZH`. **No license tag.**
  - `Magpie-Align/Magpie-Qwen2-Pro-200K-English` — the English twin.
  - **`Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1`** — **1,000,000 rows**, generated by `Qwen/Qwen2.5-72B-Instruct` with an **English** `pre_query_template`. **Measured: 99.95% English over 125,000 rows (2 of 16 shards); zero Chinese in the sample.** Do not mistake this for a Chinese resource.
- **Per-row metadata (both):** `instruct_reward` (FsfairX-LLaMA3-RM-v0.1), `input_quality` + `quality_explanation`, `difficulty`, `task_category` + `other_task_category`, `llama_guard_2`, `min_neighbor_distance`, `repeat_count`, `min_similar_uuid`, `instruction_length`, `response_length`, `language`.
- **License / 许可:** ⚠️ **No license tag on either Chinese-relevant repo.** The content is Qwen2/Qwen2.5-72B-Instruct output, so the Qwen model licence governs downstream use.
- **Quality / 质量:** `Magpie-Qwen2-Pro-200K-Chinese` is the best *natively-Chinese synthetic* set I sampled and is stable across both shards (probed at row groups 50 and 99 of shard 2). Its practical advantage over everything else in this review is the shipped filtering harness — you can cut by reward and dedup by neighbour distance without building anything. Minor defect: occasional garbled characters in synthesized prompts ("平衡因子擯至2"). | 是实测中最佳的**原生中文合成**数据，两个分片均稳定；相对全篇其他数据的实用优势在于自带过滤工具（奖励分截断 + 近邻去重），无需自建。小瑕疵：合成提问偶有乱字。

---

## 5. Instruction Back-Translation / 五、指令回译

### Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation (Zheng et al., 2024)
- **Venue / Link:** arXiv:2401.06477, https://arxiv.org/abs/2401.06477 · https://github.com/Zheng0428/COIG-Kun (Apache-2.0)
- **Data / 数据:** https://huggingface.co/datasets/m-a-p/COIG-Kun · label model https://huggingface.co/m-a-p/Kun-LabelModel
- **Size / 规模:** **367,592 rows** across three splits — `wanjuan` ~224k · `wudao` ~86.6k · `skypile` **57,339** (the split I sampled). ⚠️ The abstract claims "over a million" generated; the **released** set is ~368K.
- **Construction / 构建方式:** Humpback-style instruction back-translation with an added "Answer Polishment" stage: ~10k seed instructions train a Label Model, which labels unlabeled corpus text after perplexity/length filtering; instructions are then scored and outputs refined by a Primary Chat model. **Base/teacher is Yi (Yi-34B base per GitHub; Yi-6B in experiments) — no OpenAI model anywhere in the pipeline.**
- **Domains / 领域:** Whatever Wudao / WanJuan / SkyPile contain — academics, healthcare, literature, business; each row carries `Academic/Professional Field`, `Industry Category` and `Text type`.
- **License / 许可:** ⚠️ **No dataset license tag** (the GitHub repo is Apache-2.0, but that covers code). Inherits upstream Wudao/WanJuan/SkyPile terms.
- **Quality / 质量:** Long, information-dense, natively Chinese, and one of the very few large sets with a clean non-OpenAI provenance. Visible defect: hedging boilerplate tails ("请注意，上述信息可能不是最新的，建议直接咨询…" plus a five-item list of ways to contact a provincial education bureau) that should be stripped before training. | 长、信息密度高、原生中文，且是极少数来源上与 OpenAI 完全无关的大规模数据。可见缺陷：结尾免责样板与联系方式清单，训练前应截除。

---

## 6. Template-over-Existing-NLP-Datasets / 六、既有 NLP 数据套模板

### Firefly (流萤) — firefly-train-1.1M
- **Venue / Link:** No paper. https://github.com/yangjianxin1/Firefly
- **Data / 数据:** https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
- **Size / 规模:** Card claims **1.15M**; the live repo reports **1,649,399 rows** (798,561,871 Parquet bytes / 1,116,096,350 in memory) — the discrepancy is unexplained on the card. Card states most samples are under 600 tokens.
- **Construction / 构建方式:** **Human-written instruction templates applied over 23 common Chinese NLP tasks** — "对于每个任务，由人工书写若干种指令模板" — with the project explicitly noting it "构造了许多与中华文化相关的数据". **No ChatGPT/GPT-4 anywhere in the pipeline.** The 23-task list and per-task counts exist only inside an image (`pics/task_distribution.png`), so per-task counts are **unverified**; task types visible in my sampling and the card include NLI, Summary, Couplet (对联), AncientPoem (古诗), TextCorrection, SentimentAnalyze, KeywordRecognition, NER, MRC, lyrics, composition, Classical-Chinese translation, Jin Yong fiction.
- **License / 许可:** ⚠️ **No license.** HF returns no license tag; the GitHub API returns `"license": null`. The only stated restriction is "不得用于对社会造成危害的用途，且应当遵循基座模型的商业许可". **Effectively unlicensed.**
- **Quality / 质量:** Fully stable across offsets 0 / 2,000 / 20,000 — the task mixture at depth matches the head. Its distinctive value for a figurative-language project is the Chinese-cultural task types (couplets, classical poetry continuation, 文言文 translation) that appear in no other large open set. ⚠️ **But targets are short and extractive** — this trains task-following, not assistant behaviour; use it as a diversity component only. | 三个偏移完全稳定。对比喻性语言项目的独特价值在于其他大规模开源集所无的中华文化任务类型（对联、古诗续写、文言文翻译）。⚠️ 但目标输出短且抽取式，只宜作多样性成分。

### COIG: Chinese Open Instruction Generalist — A Preliminary Release (Zhang et al., 2023)
- **Venue / Link:** arXiv:2304.07987, https://arxiv.org/abs/2304.07987
- **Data / 数据:** https://huggingface.co/datasets/BAAI/COIG
- **Size / 规模:** Card subsets: translated **66,858** · exam **63,532** · human-value-alignment **34,471** · counterfactual-correction multi-round chat **13,653** (≈65,000 chat rounds) · Leetcode **11,737** → **card total 190,251**. ⚠️ The loader disagrees: the `Default` split is **178,246** rows and `NoTranslate` is **97,739** (275,985 across configs). Flag the mismatch.
- **Construction / 构建方式:** Five heterogeneous pipelines — (a) machine-translate-then-manually-verify (1,616 Super-NaturalInstructions task descriptions + 175 Self-Instruct seeds + 66,007 Unnatural Instructions); (b) Chinese entrance-exam questions with answer analyses; (c) human-value alignment (3,000 shared Chinese-world values plus regional sets); (d) **counterfactual correction multi-round chat built on CN-DBpedia**, 5 rounds of student-teacher role-play; (e) 2,589 scraped Leetcode problems.
- **License / 许可:** `apache-2.0`, not gated. Mixed sub-licenses: MIT for the Unnatural Instructions portion, **CC-BY-SA-4.0 for the Leetcode collection** (share-alike is a real downstream constraint). The translated subset descends from Unnatural Instructions, itself `text-davinci-002`-generated → residual OpenAI ToU exposure.
- **Quality / 质量:** The rows I sampled at offsets 0 and 2,000 are dominated by the CCMC subset, and it is the most structurally interesting data in this review: a student asserts a false premise ("玉露香梨难道不是在库尔勒市种植最多吗？") and the teacher corrects it with evidence over ~5 rounds. This is the only open Chinese data that explicitly trains **disagreeing with a wrong premise** — a known failure mode of sycophantic SFT models. Schema note: `instruction` is empty for CCMC rows; content lives in `conversations` as `{question, answer}` pairs. | 采样行以 CCMC 子集为主，是本综述结构上最有意思的数据：学生提出错误前提，教师以证据多轮纠正。这是唯一显式训练**反驳错误前提**的开源中文数据。

### COIG-PC / COIG-PC-Lite (BAAI, 2023)
- **Venue / Link:** No dedicated paper.
- **Data / 数据:** https://huggingface.co/datasets/BAAI/COIG-PC (⚠️ **gated**) · https://huggingface.co/datasets/BAAI/COIG-PC-Lite (ungated)
- **Size / 规模:** COIG-PC covers **3,339 tasks**; the card declares a `100M<n<1B` size band and no exact instance count (**unverified**); the Parquet branch lists **272 shards**. **COIG-PC-Lite: 1,078,563 rows across splits** (`train` **216,691**, plus `full`, `Top50/100/200PerTask`, `test`, `valid`), described as 200 samples per task file.
- **Construction / 构建方式:** Aggregation + normalization of "almost all available Chinese datasets in the market" by engineers from 20+ universities, with manual dedup and normalization. **Not model-distilled.** Fields: `instruction`/`input`/`output`/`split`/`task_name_in_eng`/`task_type` (major+minor)/`domain`/`other`/`filename`.
- **License / 许可:** ⚠️ **Effectively non-commercial.** HF reports `license: unknown`, and the COIG-PC access gate requires acknowledging **"I agree to use this model for non-commercial use ONLY"**; the card's default-Apache-2.0 clause is overridden by declared sub-dataset licences and by the gate.
- **Quality / 质量:** Clean, unambiguous, richly typed task data. Stable to offset 20,000, where the sampled task is Chinese-culture book identification over unpunctuated Traditional-Chinese classical text ("武王勝殷殺受立武庚以箕子歸作洪範…" → 尚書) — evidence that the deep tail carries genuinely Chinese-specific tasks. But it is task data, not chat: use for breadth, not for assistant behaviour. | 干净、明确、类型标注丰富，深层稳定且确实含中文专有任务；但属任务型数据而非对话，只宜补广度。

### Aya: an Open-Access Collection for Multilingual Instruction Tuning (Singh et al., 2024)
- **Venue / Link:** **ACL 2024**; arXiv:2402.06619, https://arxiv.org/abs/2402.06619
- **Data / 数据:** https://huggingface.co/datasets/CohereLabs/aya_dataset · https://huggingface.co/datasets/CohereLabs/aya_collection · https://huggingface.co/datasets/CohereLabs/aya_collection_language_split
- **Size / 规模 (measured, not card claims):** **Aya Dataset train = 202,362 rows total**, of which **Simplified Chinese 3,038 (1.50%)** and **Traditional Chinese 1,871 (0.92%)** — I counted the full `language` column, not a sample. The corpus is dominated by Plateau Malagasy 14,597 (7.21%), Sinhala 14,524 (7.18%), Tamil 14,133 (6.98%), Yoruba 11,758 (5.81%). **Aya Collection `chinese` config: train 58,941 / validation 7,397 / test 8,634.**
- **Construction / 构建方式:** Aya **Dataset** = original human annotations and re-annotations by a global volunteer network (fields: `inputs`, `targets`, `language`, `annotation_type`, `user_id`). Aya **Collection** = templated instructions over existing datasets plus translations, with `dataset_name`, `task_type`, `template_id` and `script: Hans/Hant` fields.
- **License / 许可:** ✅ **Apache-2.0** for both.
- **Quality / 质量:** The Aya *Dataset* is genuinely human-written and per-example high quality, but at ~4.9K Chinese rows (Simplified + Traditional) it is a seed, not a corpus. The Aya *Collection* `chinese` config has two serious defects I observed at offset 0: **near-total prompt duplication** (three consecutive rows share an identical ~200-character NER instruction differing only in the final sentence, and many rows have empty `targets`: `"Results": []`) and a **Simplified→Traditional conversion bug** in the template itself — "請識別**下麵**提供的輸入句子", where 下麵 means *noodles* and the intended word is 下面. Dedup aggressively and fix or drop that template family. | Aya *Dataset* 确为人写、单条质量高，但中文仅约 4.9 千行，只是种子；Aya *Collection* 的 `chinese` 配置在 offset 0 即暴露两处严重缺陷：**提示近乎完全重复**（大量行标准答案为空）与模板本身的**简繁转换错误**（"下麵"应为"下面"）。须强力去重并修正或剔除该模板族。

---

## 7. Reasoning Distillation (the R1 era) / 七、推理蒸馏（R1 时代）

### Chinese-Data-Distill-From-R1 — Chinese-DeepSeek-R1-Distill-data-110k (Liu, 2025)
- **Venue / Link:** No paper. https://github.com/YunwenTechnology/Chinese-Data-Distill-From-R1 · blog https://zhuanlan.zhihu.com/p/24430839729
- **Data / 数据:** https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k · ready-to-SFT variant `...-110k-SFT` (thinking + answer merged into one `output` field) · ModelScope mirror `liucong/Chinese-DeepSeek-R1-Distill-data-110k`
- **Size / 规模:** **110,000 rows** — Math 36,568 · Exam 2,432 · STEM 12,648 · General 58,352 (弱智吧、逻辑推理、小红书、知乎、Chat). Fields: `input`, `reasoning_content`, `content`, `repo_name`, `prompt_tokens_len`, `content_tokens_len`, `reasoning_content_tokens_len`, `score`.
- **Construction / 构建方式:** Chinese prompts sourced from Advanced-Math, applied_math, **`meta-math/GSM8K_zh`**, EduChat-Math, **`m-a-p/COIG-CQIA`**, `m-a-p/neo_sft_phase2` and `hfl/stem_zh_instruction`, then answered by **the full DeepSeek-R1 (671B)**, with per-row scoring (Math-Verify plus a Qwen2.5-72B judge) retained in the `score` field. The `repo_name` field preserves provenance per row.
- **License / 许可:** ✅ **Apache-2.0.** ⚠️ Content is DeepSeek-R1 output, so DeepSeek's model terms apply downstream.
- **Motivation & quality / 动机与质量:** The author's stated motivation is that "大部分开源的R1蒸馏数据集均为英文数据集" — which my sampling confirms across this whole review. It is the best native-Chinese reasoning SFT set I read, verified at offsets 0 / 2,000 / 20,000 **and** at row groups 14 and 28 of the final shard, with no degradation. The `score` field makes quality thresholding trivial. ⚠️ Caveat surfaced by deep sampling: the math slice is **prompt-translated** — a last-shard row has `repo_name: meta-math/GSM8K_zh` and reads "Courtney喜欢收集弹珠…", an English word problem in Chinese clothing. The general and STEM slices are natively Chinese. | 作者的动机——"大部分开源的R1蒸馏数据集均为英文"——正是本综述反复确认的事实。这是实测中最佳的原生中文推理 SFT 数据，深层无退化，`score` 字段使质量过滤变得简单。⚠️ 深探暴露的注意点：数学切片是**提示翻译**而来。

### AM-DeepSeek-R1-Distilled-1.4M (a-m-team, 2025)
- **Venue / Link:** a-m-team release. https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M
- **Size / 规模:** ~1.4M rows (name); **not verified by sampling** — the HF Parquet-conversion job produced **no configs**, so the viewer/Parquet path returns nothing. Chinese share **unverified**.
- **License / 许可:** ⚠️ **CC-BY-NC-4.0 — non-commercial.**
- **Quality / 质量:** **Not sampled** (no Parquet conversion). Listed so the gap is explicit rather than glossed over. | **未采样**（无 Parquet 转换）。列出以明示这一空白而非略过。

---

## 8. Curated Mega-Mixtures / 八、策展巨型混合

### Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models (BAAI, 2025)
- **Venue / Link:** arXiv:2506.11116, https://arxiv.org/abs/2506.11116 · follow-up "Infinity Instruct Subject" arXiv:2507.06968
- **Data / 数据:** https://huggingface.co/datasets/BAAI/Infinity-Instruct — ⚠️ **`gated: auto`; I could not read a single row.** Configs listed: `0625`, `3M`, `7M`, `7M_core`, `7M_domains`, `Gen`.
- **Size / 规模:** **InfInstruct-F-7.4M = 7,449,106** foundational instructions curated from over 100M candidates; **InfInstruct-G-1.5M = 1,456,927** chat instructions synthesized in a second phase; `3M` = 3,463,473; `0625` chat = 659,808. **Per-language breakdown is not published — the Chinese share is unverified.**
- **Construction / 构建方式:** Two phases — (1) hybrid data *selection* over a >100M pool (Flan, OpenHermes, code and math seeds); (2) instruction *evolution* along breadth / depth / difficulty / complexity plus diagnostic filtering.
- **License / 许可:** **CC-BY-SA-4.0** (share-alike — copyleft for downstream mixes).
- **Quality / 质量:** **Not sampled — gated.** ✅ **Workaround that I did sample:** `Mxode/Chinese-Instruct` config `infinity-instruct` redistributes **386,426 rows** of Infinity-Instruct's Chinese single-turn data, re-language-detected because "其自带的语言元数据仍有误" (its own language metadata is wrong). Those rows read as fluent, well-organized general Chinese instruction data (drama scripts, economics explainers) with no visible translationese. | **未采样——受限。** ✅ 可行替代（已实测）：`Mxode/Chinese-Instruct` 的 `infinity-instruct` 子集转发了 386,426 行中文单轮数据，并因原库语言元数据有误而重做了语言检测；实测流畅、组织良好、无明显翻译腔。

### Mxode/Chinese-Instruct — 中文指令微调数据集
- **Venue / Link:** No paper. https://github.com/Mxoder/Maxs-Awesome-Datasets
- **Data / 数据:** https://huggingface.co/datasets/Mxode/Chinese-Instruct · simplified variant https://huggingface.co/datasets/Mxode/Chinese-Instruct-Lite
- **Size / 规模 (verified per-config `num_rows`):** **4,845,389 total** — `firefly` 1,061,823 · `chinese-medical` 998,387 · `Chinese-QA-AFAF` 676,685 · `magpie` 542,044 · `infinity-instruct` 386,426 · `industryinstruction` 358,582 · `stem_zh` 214,646 · `sof-c-zh` 198,335 · `chinese-reasoning-distil` 179,037 · `dpsk-r1-distil` 100,331 · `neo_sft_phase2` 58,549 · `disc-law` 51,463 · `coig-cqia` 10,306 · `psycho-10k-dpsk-r1` 8,775.
- **Construction / 构建方式:** Not a raw aggregation — each subset is reprocessed. `firefly` instructions were rephrased and **re-answered with DeepSeek-V2.5**; `stem_zh` answers were re-synthesized and hallucinations removed; `magpie` instructions were sampled from Magpie then **rewritten and filtered with GPT-4o and answered with GPT-4o-mini**; `infinity-instruct` and `neo_sft_phase2` were **re-language-detected** and heuristically filtered; `coig-cqia` keeps only human-verified rows; `dpsk-r1-distil` is score-filtered and keeps only the final answer (no CoT).
- **License / 许可:** ✅ **CC-BY-SA-4.0** — notable because most of its upstreams (Firefly, COIG-CQIA) declare **no** licence at all. ⚠️ Share-alike is copyleft; check compatibility with your release plan.
- **Quality / 质量:** The most practical single entry point for a 2025-26 Chinese SFT mix. `stem_zh` and `firefly` sampled excellently; `dpsk-r1-distil` inherits the R1 data's quality. ⚠️ **The `magpie` config is the weak link** — GPT-4o-rewritten from English Magpie, it reads as translationese with non-Chinese context ("在新加坡…向IRAS提交税务申报表"), and at offset 20,000 a prompt about "我妹妹" is answered about "你姐姐", a slip that only occurs when translating English `sister`. Also note `neo_sft_phase2` **only survives here** — the original `m-a-p/neo_sft_phase2` now 404s. | 是当下做中文 SFT 混合最实用的单一入口。`stem_zh`/`firefly` 实测优秀。⚠️ `magpie` 子集是短板（翻译腔、姐妹错位）。另注意 `neo_sft_phase2` **仅在此处留存**，原库已 404。

### deepctrl-sft-data — 匠数科技大模型 SFT 数据集
- **Venue / Link:** No paper.
- **Data / 数据:** https://huggingface.co/datasets/deepctrl/deepctrl-sft-data (**HF API returned 404 to my token**) · https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data · https://opendatalab.com/OpenDataLab/deepctrl-sft-data
- **Size / 规模:** Mirrors report **Chinese 11,381,621 + English 2,767,403** (OpenDataLab's own summary rounds these to "10M Chinese / 2M English"). ⚠️ These digits are **second-hand** — I could not read the primary card.
- **Construction / 构建方式:** Aggregation of existing open datasets (attributed sources include BelleGroup, LinkSoul, BAAI and TigerResearch) with format unification, cleaning and content review. 50 task categories; 12 fields including `history` (multi-turn) and `num_utter`.
- **License / 许可:** Apache-2.0 per the mirrors. ⚠️ **Transitively broken:** because BELLE is a documented input, this dataset inherits BELLE's research-only + OpenAI-ToU constraints regardless of its own permissive tag.
- **Quality / 质量:** **Not sampled** (inaccessible). Note the description itself concedes the Chinese file "includes some foreign language data (because there are some language translation tasks)". | **未采样**（不可访问）。其自述亦承认中文文件包含部分外语数据。

### Tulu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024)
- **Venue / Link:** arXiv:2411.15124, https://arxiv.org/abs/2411.15124
- **Data / 数据:** https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- **Size / 规模:** **939,343 rows.** Each row carries a `source` field naming its origin (`ai2-adapt-dev/oasst1_converted`, Persona-hub sets, WildGuard, math and code sets…).
- **Construction / 构建方式:** A carefully documented mixture combining public sets with synthetically generated persona-driven data, followed by DPO and RLVR stages. The most transparent open post-training recipe available.
- **License / 许可:** ✅ **ODC-BY.**
- **Measured Chinese share / 实测中文占比:** A CJK-density scan over **2 of 6 shards = 313,115 rows** found **6,728 Chinese rows (2.15%)**, extrapolating to roughly **20K** over the full 939,343. Broken down by `source`: `ai2-adapt-dev/tulu_v3.9_wildchat_100k` **6,362 / 100,000 (6.4%)**, `oasst1_converted` 248 / 7,131, `flan_v2_converted` 105 / 89,982, `personahub_math_v5_regen_149960` 13 / 106,262. **Tulu 3's Chinese is essentially WildChat's Chinese, subsampled.**
- **Quality / 质量:** Excellent and the most transparent open post-training recipe available, but **English-centric** — offsets 0 / 2,000 / 20,000 returned English and Spanish (OASST1-converted) rows. Its role in a Chinese SFT mix is as an **English anchor** against the forgetting a zh-only SFT stage induces on a Qwen base, not as a zh resource; and if you want its Chinese, take it from WildChat directly at full resolution. | 对 **6 个分片中的 2 个、共 313,115 行**做中日韩字符密度扫描，得 **6,728 行中文（2.15%）**，全量外推约 **2 万行**；按 `source` 拆解见上——**Tulu 3 的中文本质上就是被下采样的 WildChat 中文**。质量优秀且是最透明的公开后训练配方，但以英文为中心；在中文混合中作**英文锚点**，若要其中文则应直接从 WildChat 全量取用。

---

## 9. Preference / DPO / RLHF Data for Chinese (secondary section) / 九、中文偏好、DPO 与 RLHF 数据（次要章节）

> **Summary of what I found by sampling / 实测小结:** The open Chinese preference landscape is the weakest part of the ecosystem. The two best-known "modern" preference sets — `BAAI/Infinity-Preference` and `Skywork/Skywork-Reward-Preference-80K-v0.2` — are **English in every row I sampled**, despite being published by Chinese labs. The two widely-cited Chinese ones are **machine translations of Anthropic HH-RLHF with visible MT damage**. Only COIG-P is both large and natively Chinese, and I could not sample it. | 开源中文偏好数据是整个生态最薄弱的一环：两个最知名的"现代"偏好集虽由中国实验室发布，但我采样的每一行都是英文；两个被广泛引用的中文集是 Anthropic HH-RLHF 的机器翻译且损伤明显。唯一既大规模又原生中文的是 COIG-P，而它未能采样。

### COIG-P: A High-Quality and Large-Scale Chinese Preference Dataset for Alignment with Human Values (2025)
- **Venue / Link:** arXiv:2504.05535, https://arxiv.org/abs/2504.05535 · https://github.com/multimodal-art-projection/COIG-P
- **Size / 规模:** **1,009k preference pairs** over 6 domains (Chat, Code, Math, Logic, Novel, Role), from 92k filtered queries.
- **Construction / 构建方式:** 15 LLMs generate and score chosen/rejected pairs; **no human intervention** in the loop.
- **License / 许可:** ⚠️ **Unverified.**
- **Quality / 质量:** **Not sampled** in this review. On paper it is the only open Chinese preference resource at a scale that matters; verify by sampling before committing to it. | 本综述**未采样**。就规模而言，它是唯一量级足够的开源中文偏好资源；投入前请先自行抽样验证。

### BAAI/Infinity-Preference
- **Data / 数据:** https://huggingface.co/datasets/BAAI/Infinity-Preference · **59,338 train rows** · ✅ **Apache-2.0**
- **Quality / 质量:** ⚠️ **English.** Every row I sampled at offset 0 is English (PGP/GPG key generation on Ubuntu; propositional-logic reasoning), with a `task_category` field. Chosen and rejected are near-identical long technical answers — a fine-grained preference signal, but not a Chinese one. | ⚠️ **英文**。offset 0 采样行全为英文，chosen 与 rejected 是近乎相同的长技术答案：偏好信号细粒度，但不是中文的。

### Skywork/Skywork-Reward-Preference-80K-v0.2
- **Data / 数据:** https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2 · **77,016 rows** · license tag absent
- **Quality / 质量:** ⚠️ **English.** Sampled rows carry `source: magpie_ultra` and are English coding problems (FASTA/GFF parsing, factorials). Not a Chinese resource. | ⚠️ **英文**，采样行 `source: magpie_ultra`，为英文编程题。

### dikw/hh_rlhf_cn
- **Data / 数据:** https://huggingface.co/datasets/dikw/hh_rlhf_cn · **344,317 train rows** · license tag `llama2`
- **Construction / 构建方式:** Machine translation of Anthropic HH-RLHF (helpful + harmless) into Chinese.
- **Quality / 质量:** ⚠️ **Poor at every offset sampled (0 / 2,000 / 20,000).** Degenerate repetition ("大便、大便、大便…"), doubled words ("并不同时同时生活"), broken punctuation spacing. The underlying Anthropic dialogues are also deliberately adversarial, so the Chinese reads as both toxic and broken. | ⚠️ **各偏移一致地差**：退化重复、叠词、标点错乱；底层 Anthropic 对话本就带对抗性，中译后既有毒又破碎。

### beyond/rlhf-reward-single-round-trans_chinese
- **Data / 数据:** https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese · **19,862 train rows** · ⚠️ **no license**
- **Quality / 质量:** ⚠️ **Poor.** Untranslated English leaks into the Chinese ("你对\"indoors\"是什么理解？"), and in several sampled pairs **both** `chosen` and `rejected` are weak clarifying questions rather than answers, so the preference signal is close to noise. | ⚠️ **差**：英文残留漏入中文；多组采样中 chosen 与 rejected **双方**都只是弱化的澄清提问，偏好信号接近噪声。

### wenbopan/Chinese-dpo-pairs
- **Data / 数据:** https://huggingface.co/datasets/wenbopan/Chinese-dpo-pairs · **10,735 rows** · ✅ **MIT**
- **Quality / 质量:** Small; the sampled prompts are translated English task-definition instructions (Amazon food-review polarity checking, peanut-butter nutrition), i.e. Chinese wrapping around English task content. Usable as a small DPO seed, not as a Chinese alignment signal. | 规模小；采样提示为英译的任务定义型指令（亚马逊食评极性判断等），即中文外壳包裹英文任务内容。可作小型 DPO 种子，不足以作中文对齐信号。

### Other Chinese preference resources (not sampled) / 其他中文偏好资源（未采样）
- **CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility** — arXiv:2307.09705, https://github.com/X-PLUG/CValues · ~145K · safety and responsibility value alignment.
- **`liyucheng/zhihu_rlhf_3k`** — ~3K+ Chinese preference pairs derived from Zhihu upvote signals.
- **`Hello-SimpleAI/HC3-Chinese` repurposed** — 12,853 questions each with human and ChatGPT answers; the cleanest way to get a *human-preferred* Chinese signal at small scale, since the "chosen" side is genuinely human-written.

---

## 10. Model Technical Reports — What They Say, and What They Released / 十、模型技术报告：说了什么，公开了什么

> **The single most important structural fact in this review:** every Chinese frontier lab documents its SFT mixture qualitatively and releases weights, and **not one of them released the instruction data**. Hunyuan-Large does not release its >1M synthetic SFT set; TeleChat releases 1 TB of *pretraining* text but not its 100K SFT samples; Skywork releases SkyPile (pretraining) and OR1 RL prompts (English math/code) but no Chinese SFT. The open Chinese SFT ecosystem is therefore entirely community-built. | **本综述最重要的结构性事实**：所有中文前沿实验室都定性描述了其 SFT 混合并公开了权重，但**没有一家公开指令数据**。因此开源中文 SFT 生态完全由社区构建。

| Model / 模型 | arXiv | Reported SFT size / 报告的 SFT 规模 | How built / 构建方式 | Released? / 是否公开 | zh:en ratio |
|---|---|---|---|---|---|
| Qwen | [2309.16609](https://arxiv.org/abs/2309.16609) | **No number given** in the SFT section | Human-style annotated conversations in multiple styles, ChatML format; explicitly beyond self-instruct. Seq 2048, batch 128, 4000 steps, LR 2e-6 | **No** | Not stated |
| Qwen2 | [2407.10671](https://arxiv.org/abs/2407.10671) | **>500,000 examples** | Collaborative annotation (InsTag ontology, diversity/complexity selection, self-evolution, human ranking) + automated synthesis (rejection sampling for math, execution feedback for code, data repurposing from literary works, constitutional feedback). 2 epochs, seq 32,768 | **No** | Not stated |
| Qwen2.5 | [2412.15115](https://arxiv.org/abs/2412.15115) | **>1,000,000 SFT examples**; DPO **~150,000 pairs** | 9 targeted areas incl. long-sequence generation (8,192-token outputs), math CoT from Qwen2.5-Math, code from Qwen2.5-Coder (~40 languages), **70,000 new logical-reasoning queries**, **cross-lingual transfer via translation**, hundreds of system prompts, multi-agent response filtering. 2 epochs, seq 32,768 | **No** | Not stated |
| Qwen3 | [2505.09388](https://arxiv.org/abs/2505.09388) | Cold-start and Thinking-Mode-Fusion SFT counts **not disclosed**; Reasoning RL uses **3,995 query-verifier pairs** | 4 stages: Long-CoT cold start (QwQ-32B candidates, filtered) → Reasoning RL (GRPO) → Thinking-Mode Fusion → General RL over 20+ tasks. Strong-to-weak distillation, off-policy then on-policy KL | **No** | Pretraining spans 119 languages/dialects, 36T tokens; no SFT-level ratio |
| Yi | [2403.04652](https://arxiv.org/abs/2403.04652) | **<10K multi-turn dialogue pairs** | LIMA-style quality-over-quantity; every entry "constructed and polished over multiple iterations and from user feedback"; WizardLM-style compound instructions, CoT patterns, hallucination screening. LR 1e-5, bs 64, seq 4096, 300 steps, NEFTune | **No** | "Bilingual"; no ratio |
| InternLM2 | [2403.17297](https://arxiv.org/abs/2403.17297) | **10,000,000 instruction instances**; COOL RLHF on **up to 2.4M binarized preference pairs** | Screened for helpfulness/harmlessness; general conversation, NLP tasks, math, code, function calls. ChatML, 1 epoch, AdamW LR 4e-5 | **No** (an SFT-only *checkpoint*, `InternLM2-Chat-SFT`, was released) | Not stated |
| DeepSeek LLM | [2401.02954](https://arxiv.org/abs/2401.02954) | **~1.5M instances = 1.2M helpful + 300K safety** | Helpful split: 31.2% general language, 46.6% math, 22.2% coding. 7B: 4 epochs @1e-5; 67B: 2 epochs @5e-6; DPO 1 epoch @5e-6 | **No** | "English and Chinese"; no numeric ratio |
| DeepSeek-V3 | [2412.19437](https://arxiv.org/abs/2412.19437) | **1.5M instances** | Reasoning data from an internal DeepSeek-R1 + rejection sampling; non-reasoning from DeepSeek-V2.5 with human verification. 2 epochs | **No** | Not stated |
| DeepSeek-R1 | [2501.12948](https://arxiv.org/abs/2501.12948) | **~800K = ~600K reasoning + ~200K non-reasoning**; plus "thousands" of cold-start CoT | Cold start: few-shot long-CoT prompting + reformatted R1-Zero outputs + human post-processing. Reasoning: rejection sampling from an RL checkpoint with rule-based and generative rewards | **No** — but the community reproduced the Chinese side as `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` (see §7) | Not stated |
| Baichuan 2 | [2309.10305](https://arxiv.org/abs/2309.10305) | **>100K SFT samples** | Human annotators with Claude-style principles; cross-validation QC where an authoritative annotator audits batches | **No** | Not stated |
| ChatGLM / GLM-4 | [2406.12793](https://arxiv.org/abs/2406.12793) | **Not disclosed** anywhere in the report | Early: prompt-response pairs annotated by the developers themselves. Later: in-house annotation plus proprietary third-party data. Explicitly argues "authentic human prompts and interactions instead of template-based or model-generated responses are vital" | **No** | Not stated |
| MiniCPM | [2404.06395](https://arxiv.org/abs/2404.06395) | **~6 billion tokens** of SFT (given in tokens, not examples) | Decay-stage mixture of UltraChat, SlimOrca, OssInstruct, EvolInstruct plus proprietary LeetCode and K-12 material; DPO on UltraFeedback plus proprietary code/math preferences | **No** | The named public components are overwhelmingly **English** |
| Skywork | [2310.19341](https://arxiv.org/abs/2310.19341) | Not covered — "we focus on the development of the base model" | n/a | **SFT no.** Released **SkyPile-150B** — *pretraining*, not SFT | Full training corpus: en 49.8% / zh 39.6% / code 8.0% / other 2.4% — **pretraining, do not miscite as SFT** |
| Skywork-OR1 | [2505.22312](https://arxiv.org/abs/2505.22312) | RL prompt set, not SFT | Curated from 7 open datasets with model-aware difficulty estimation and contamination dedup | **RL data yes**: `Skywork/Skywork-OR1-RL-Data`, **119,112 examples** (105K math + 14.1K code); licence not stated | **Problems are in English** |
| TeleChat | [2401.03804](https://arxiv.org/abs/2401.03804) | **>100,000 SFT samples** | Internal + contracted annotators, all native Chinese speakers; trial-annotation selection; two-stage review on fluency, helpfulness, truthfulness, harmlessness | **SFT no** (1 TB of *pretraining* text was released) | Chinese-native annotation; no ratio |
| Hunyuan-Large | [2411.02265](https://arxiv.org/abs/2411.02265) | **>1,000,000 SFT examples** | 4-step synthetic pipeline: instruction generation from web/QA/code/book seeds → instruction evolution → response generation by specialized models → filtering via a critique model and self-consistency | **No** — only the PenguinScrolls long-context benchmark | "Primarily Chinese and English"; no ratio |

**Reading of the table / 表格解读:** Reported SFT sizes span three orders of magnitude — Yi's **<10K** to InternLM2's **10M** — with no correlation to claimed quality, which is the clearest evidence available that *composition beats volume* for Chinese SFT. And **Chinese/English ratios are essentially never disclosed**: only DeepSeek LLM ("English and Chinese") and Hunyuan ("primarily Chinese and English") say anything, and neither gives numbers. | 报告的 SFT 规模跨越三个数量级（Yi 的 **<1 万** 到 InternLM2 的 **1000 万**），且与声称的质量无相关性——这是"配比重于规模"最直接的证据。而**中英配比几乎从不披露**。

---

## 11. Benchmarks & Evaluation for Chinese Instruction Following / 十一、中文指令遵循的评测基准

These are **evaluation** resources, not training data, but they are what you should score the SFT stage on. | 以下为**评测**资源而非训练数据，但应作为 SFT 阶段的打分依据。

- **AlignBench** — arXiv:2311.18743, https://arxiv.org/abs/2311.18743 (ACL 2024) · https://github.com/THUDM/AlignBench · **683 real-scenario Chinese queries** with human-verified references, 8 categories, rule-calibrated multi-dimensional LLM-as-Judge with CoT. Adopted by ChatGLM, Qwen, DeepSeek, Yi and Baichuan — the de-facto Chinese alignment benchmark.
- **CIF-Bench** — arXiv:2402.13109, https://arxiv.org/abs/2402.13109 (Findings of ACL 2024) · https://github.com/yizhilll/CIF-Bench · **150 tasks, 15,000 native-speaker-authored input-output pairs**, 20 categories, ×3 diversified instructions → 45,000 instances. ⚠️ **Only half is public** (contamination control).
- **CHiSafetyBench** — arXiv:2406.10311, https://arxiv.org/abs/2406.10311 · hierarchical Chinese safety taxonomy: 5 risk areas, 31 categories, **1,861 multiple-choice questions** plus QA tasks. Data at https://github.com/UnicomAI/DataSet/tree/main/TestData/Safety. No official HF dataset; licence unspecified.
- **SuperCLUE** — arXiv:2307.15020, https://arxiv.org/abs/2307.15020 · https://github.com/CLUEbenchmark/SuperCLUE · three sub-tasks (CArena user-battle ratings, OPEN open-ended single/multi-turn, CLOSE closed-ended). ⚠️ Which portions are open vs. held out is **not cleanly documented**.
- **COIG-Writer** — arXiv:2510.14763 · **1,665 triplets** (reverse-engineered prompt + creative reasoning + final text) across **51 genres**; reports an optimal ratio of 1 creative sample per 12 general samples — a directly actionable mixing finding if creative Chinese writing matters to you.

---

## 12. Comparison Table / 对比表

Sizes are **verified row counts** (Parquet metadata or `datasets-server` `/size`) unless marked *card*. "Sampled" = I read real rows from it. | 除标注 *card* 者外，规模均为**实测行数**。"Sampled" 表示我实际读取了真实数据行。

| # | Dataset / 数据集 | Type / 类型 | Size (rows) / 规模 | Construction / 构建 | Teacher / 教师模型 | zh share / 中文占比 | License / 许可 | Sampled? |
|---|---|---|---|---|---|---|---|---|
| 1 | `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` | reasoning SFT | **110,000** | distilled | DeepSeek-R1 671B | 100% | ✅ Apache-2.0 | ✅ deep |
| 2 | `allenai/WildChat-1M` | real-user logs | **837,989** | real prompts + model answers | gpt-3.5-turbo / gpt-4 | **23.27% measured** | ⚠️ ODC-BY + OpenAI ToU | ✅ deep |
| 3 | `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese` | general chat | **200,000** | prompt-free self-synthesis | Qwen2-72B-Instruct | 100% | ⚠️ none | ✅ deep |
| 4 | `m-a-p/COIG-CQIA` | human-written | **44,694** (card 48,384) | human web content + manual verify | none | 100% | ⚠️ **none declared** | ✅ 5 configs |
| 5 | `Mxode/Chinese-Instruct` | mega-mixture | **4,845,389** | curated aggregation + re-answering | DeepSeek-V2.5, GPT-4o/-mini | 100% | ✅ CC-BY-SA-4.0 | ✅ 5 configs |
| 6 | `BelleGroup/train_3.5M_CN` | multi-turn chat | **3,606,402** | self-instruct | davinci-003 / gpt-3.5-turbo | 100% | ⚠️ GPL-3.0 tag / research-only prose | ✅ |
| 7 | `YeungNLP/firefly-train-1.1M` | NLP-task templates | **1,649,399** (card 1.15M) | human templates over 23 zh tasks | **none** | 100% | ⚠️ **none** | ✅ |
| 8 | `wangrui6/Zhihu-KOL` | scraped human | **1,006,218** | scraped Zhihu | none | 100% | ⚠️ **none (scraped)** | ✅ |
| 9 | `BAAI/COIG` | mixed 5 pipelines | **178,246** `Default` (card 190,251) | translate+verify / exam / CN-DBpedia | davinci-002 (translated subset) | 100% | ✅ Apache-2.0 (mixed sub-licences) | ✅ |
| 10 | `BAAI/COIG-PC-Lite` | NLP-task collection | **1,078,563** (train 216,691) | aggregation + normalization | none | 100% | ⚠️ unknown; parent gate = non-commercial | ✅ |
| 11 | `m-a-p/COIG-Kun` | back-translation | **367,592** | instruction back-translation | **Yi (no OpenAI)** | 100% | ⚠️ none | ✅ |
| 12 | `TigerResearch/sft_zh` | general SFT | **530,705** | Alpaca-style generation | unnamed (unverified) | 100% | ✅ Apache-2.0 (mixed upstream) | ✅ |
| 13 | `YeungNLP/moss-003-sft-data` | multi-turn chat | **670,948** (official 1,074,551) | real beta-user prompts + distillation | gpt-3.5-turbo | **~50%** | ⚠️ CC-BY-4.0 tag / CC-BY-NC per README | ✅ |
| 14 | `Hello-SimpleAI/HC3-Chinese` | human vs ChatGPT | **12,853** questions | existing human QA + ChatGPT answers | ChatGPT | 100% | ⚠️ CC-BY-SA-4.0 (medicine split CC-BY-NC) | ✅ |
| 15 | `Azure99/blossom-v6-sft-stage1` | curated mixture | **149,750** | multi-source recipe | mixed | ~50% | ✅ Apache-2.0 | ✅ |
| 16 | `llm-wizard/alpaca-gpt4-data-zh` | general SFT | **48,818** | translated prompts + GPT-4 answers | GPT-4 | 100% | ⚠️ card CC-BY-4.0 / **upstream CC-BY-NC-4.0** | ✅ |
| 17 | `shibing624/alpaca-zh` | **duplicate of #16** | **48,818** | — | GPT-4 | 100% | CC-BY-4.0 | ✅ (identical rows) |
| 18 | `FreedomIntelligence/Evol-Instruct-Chinese-GPT4` | evol-instruct | **70,000** | translated Evol-Instruct + GPT-4 | GPT-4 | 100% | ⚠️ **none** | ✅ |
| 19 | `silk-road/Wizard-LM-Chinese-instruct-evol` | evol-instruct | **70,000** | translated Evol-Instruct | GPT | broken | CC-BY-4.0 | ✅ (**Poor**) |
| 20 | `shibing624/sharegpt_gpt4` | ShareGPT | **103,415** | filtered ShareGPT + MT | GPT-4 | **~37%, deep offsets only** | ⚠️ CC-BY-4.0 + OpenAI ToU | ✅ 6 offsets |
| 21 | `CohereLabs/aya_dataset` | human multilingual | **202,362** train | human annotation | none | **1.50% Hans + 0.92% Hant** | ✅ Apache-2.0 | ✅ full count |
| 22 | `CohereLabs/aya_collection_language_split` (`chinese`) | templated | **58,941** train | templates over existing sets | none | 100% (of that config) | ✅ Apache-2.0 | ✅ |
| 23 | `Chinese-Vicuna/instruct_chat_50k.jsonl` | multi-turn | **51,584** | mixed | GPT | 100% (**Traditional**) | ✅ Apache-2.0 | ✅ |
| 24 | `BelleGroup/multiturn_chat_0.8M` | multi-turn (flattened) | **831,036** | self-instruct | gpt-3.5-turbo | 100% | ⚠️ GPL-3.0 / research-only | ✅ (**format unusable**) |
| 25 | `BelleGroup/school_math_0.25M` | math | **248,481** | self-instruct | gpt-3.5-turbo | 100% | ⚠️ GPL-3.0 / research-only | ✅ (**wrong solutions**) |
| 26 | `BelleGroup/train_0.5M_CN` / `train_1M_CN` / `train_2M_CN` / `generated_chat_0.4M` | general SFT | **519,255 / 917,424 / 2,000,000 / 396,004** | self-instruct | davinci-003 / gpt-3.5-turbo | 100% | ⚠️ GPL-3.0 / research-only | verified counts |
| 27 | `allenai/tulu-3-sft-mixture` | English anchor | **939,343** | curated + persona synthesis | mixed | **2.15% measured** (≈20K; from its WildChat subsample) | ✅ ODC-BY | ✅ CJK scan |
| 28 | `Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1` | general chat | **1,000,000** | prompt-free self-synthesis | Qwen2.5-72B-Instruct | **99.95% EN measured** | ⚠️ none | ✅ full count |
| 29 | `dikw/hh_rlhf_cn` | preference | **344,317** train | MT of Anthropic HH | — | 100% (broken MT) | tag `llama2` | ✅ (**Poor**) |
| 30 | `beyond/rlhf-reward-single-round-trans_chinese` | preference | **19,862** train | MT of Anthropic HH | — | 100% (leaky MT) | ⚠️ **none** | ✅ (**Poor**) |
| 31 | `wenbopan/Chinese-dpo-pairs` | preference | **10,735** | translated task instructions | — | 100% | ✅ MIT | ✅ |
| 32 | `BAAI/Infinity-Preference` | preference | **59,338** train | LLM-scored pairs | mixed | ~0% in sample | ✅ Apache-2.0 | ✅ (**EN**) |
| 33 | `Skywork/Skywork-Reward-Preference-80K-v0.2` | preference | **77,016** | curated pairs | mixed | ~0% in sample | not stated | ✅ (**EN**) |
| 34 | `BAAI/Infinity-Instruct` | mega-mixture | **7,449,106** (`7M`, *card/paper*) | selection + evolution | mixed | unverified | ⚠️ CC-BY-SA-4.0 | ❌ **gated** |
| 35 | `lmsys/lmsys-chat-1m` | real-user logs | 1,000,000 *card* | Chatbot Arena traffic | 25 LLMs | unverified | custom | ❌ **gated** |
| 36 | `BAAI/COIG-PC` | NLP-task collection | 3,339 tasks *card* | aggregation | none | 100% | ⚠️ gate = **non-commercial** | ❌ **gated** |
| 37 | `BAAI/IndustryInstruction` | domain SFT | zh slice **358,582** (via Mxode) | Doc2QA / Topic2QA synthesis | mixed | 100% | ✅ Apache-2.0 | ❌ **gated** |
| 38 | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | reasoning SFT | ~1.4M *card* | R1 distillation | DeepSeek-R1 | unverified | ⚠️ **CC-BY-NC-4.0** | ❌ no Parquet |
| 39 | `shareAI/ShareGPT-Chinese-English-90k` | ShareGPT | ~90K *card* | user-shared + MT | ChatGPT | ~50% | ⚠️ Apache-2.0 tag over ChatGPT output | ❌ viewer job fails |
| 40 | `deepctrl/deepctrl-sft-data` | mega-mixture | zh 11,381,621 + en 2,767,403 *mirror* | aggregation | inherited | ~80% file share | ⚠️ Apache-2.0 claim, BELLE-tainted | ❌ 404 |
| 41 | `BAAI/OL-CC` | human crowdsourced | ~10k *card* | fully human-written | **none** | 100% | ⚠️ CC-BY-NC 4.0 academic / paid commercial | ❌ **gated** |
| 42 | `m-a-p/neo_sft_phase2` | general SFT | zh slice **58,549** (via Mxode) | — | — | 100% (zh slice) | — | ❌ **404, repo removed** |
| 43 | `wenge-research/yayi2_sft_data` | — | — | — | — | — | — | ❌ **does not exist** |
| 44 | COIG-P | preference | **1,009k pairs** *paper* | 15 LLMs generate + score | 15 LLMs | 100% | ⚠️ unverified | ❌ not sampled |

---

## 13. Cross-Cutting Flags / 横向风险提示

**⚠️ Non-commercial or research-only** / **非商用或仅限研究:** the entire BELLE family (prose, despite GPL-3.0 tags), COIG-PC and COIG-PC-Lite (gate checkbox), all three MOSS releases (README CC-BY-NC, despite `cc-by-4.0` HF tags), `alpaca-gpt4-data-zh` (upstream CC-BY-NC-4.0, despite the `cc-by-4.0` card tag), `alpaca_data_zh_51k` (source repo: 严禁用于商业用途, despite an `apache-2.0` re-host tag), YAYI 2 data, `AM-DeepSeek-R1-Distilled-1.4M` (CC-BY-NC-4.0), BAAI/OL-CC (academic NC), and HC3's `medicine` split (CC-BY-NC upstream). **In five of these cases the HF licence tag is more permissive than the upstream terms — always cite the upstream.**

**⚠️ No licence declared at all** (highest reuse risk) / **完全未声明许可（复用风险最高）:** `YeungNLP/firefly-train-1.1M`, `wangrui6/Zhihu-KOL` (scraped Zhihu content), `m-a-p/COIG-CQIA` ("[More Information Needed]"), `m-a-p/COIG-Kun` (dataset side), `FreedomIntelligence/Evol-Instruct-Chinese-GPT4`, `beyond/rlhf-reward-single-round-trans_chinese`, and **both Chinese-relevant Magpie repos**.

**⚠️ OpenAI/GPT-output-derived** (ToU competing-model restrictions apply regardless of the stated licence) / **源自 OpenAI 模型输出（无论标称许可，均受其使用条款约束）:** all BELLE, MOSS 002 + 003, `alpaca_data_zh_51k`, `alpaca-gpt4-data-zh`, HC3's ChatGPT answers, both ShareGPT variants, WildChat's responses, COIG's Unnatural-Instructions-derived translated subset, TigerBot's `alpaca-zh-0.5m` (probable, unverified), `Mxode/Chinese-Instruct`'s `magpie` config (GPT-4o/-mini), and `deepctrl` transitively via BELLE.

**✅ Not GPT-derived** (clean on that axis) / **非 GPT 派生（该维度上干净）:** `firefly-train-1.1M` (human templates), `COIG-CQIA` (human answers), `COIG-Kun` (Yi-based), `Zhihu-KOL` (scraped human), `COIG-PC`/`COIG-PC-Lite` (aggregation), `Aya Dataset` (human annotation), `BAAI/OL-CC` (crowdsourced human), and the **Magpie Chinese release** (Qwen2-72B-Instruct, so Qwen terms rather than OpenAI terms), plus the human half of HC3.

**⚠️ Card-vs-viewer size mismatches to cite carefully** / **卡片与实测规模不一致，引用时须小心:** Firefly (1.15M claimed / **1,649,399** actual) · BELLE `train_1M_CN` ("1M" / **917,424**) · COIG (190,251 card / **178,246** loader) · COIG-CQIA (48,384 card / **44,694** configs) · COIG-Kun (paper "over a million" / **367,592** released) · `alpaca-gpt4-data-zh` ("52K" / **48,818**).

**⚠️ Not actually majority-Chinese despite the name or reputation** / **名不副实——实际并非以中文为主:** MOSS-002 (50.7% zh by row) · MOSS-003 (~50/50) · `shareAI/ShareGPT-Chinese-English-90k` (parallel bilingual; the Chinese half is translated) · `shibing624/sharegpt_gpt4` (**offsets 0–60,000 are entirely English**) · `Azure99/blossom-*` (~1:1 by design) · `Magpie-Qwen2.5-Pro-1M-v0.1` (**99.95% EN measured**) · `CohereLabs/aya_dataset` (**2.4% Chinese including Traditional**) · `deepctrl` (the "Chinese" file admits it contains foreign-language translation-task text).

**⚠️ Exact-duplicate repos** / **完全重复的库:** `shibing624/alpaca-zh` and `llm-wizard/alpaca-gpt4-data-zh` are the same 48,818 rows — I verified offset-0 rows are byte-identical. Do not mix both.

---

## 14. References / 参考文献

1. Ji et al. *Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases* (BELLE). arXiv:2303.14742. https://arxiv.org/abs/2303.14742
2. Ji et al. *Towards Better Instruction Following Language Models for Chinese*. arXiv:2304.07854. https://arxiv.org/abs/2304.07854
3. Zhang et al. *Chinese Open Instruction Generalist: A Preliminary Release* (COIG). arXiv:2304.07987. https://arxiv.org/abs/2304.07987
4. Bai et al. *COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning*. Findings of NAACL 2025. arXiv:2403.18058. https://arxiv.org/abs/2403.18058
5. Zheng et al. *Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation* (COIG-Kun). arXiv:2401.06477. https://arxiv.org/abs/2401.06477
6. *COIG-P: A High-Quality and Large-Scale Chinese Preference Dataset for Alignment with Human Values*. arXiv:2504.05535. https://arxiv.org/abs/2504.05535
7. *COIG-Writer*. arXiv:2510.14763. https://arxiv.org/abs/2510.14763
8. Sun et al. *MOSS: An Open Conversational Large Language Model*. Machine Intelligence Research 21(5): 888–905, 2024. DOI 10.1007/s11633-024-1502-8
9. Guo et al. *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (HC3). arXiv:2301.07597. https://arxiv.org/abs/2301.07597
10. Cui, Yang, Yao. *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca*. arXiv:2304.08177. https://arxiv.org/abs/2304.08177
11. Peng et al. *Instruction Tuning with GPT-4*. arXiv:2304.03277. https://arxiv.org/abs/2304.03277
12. Xu et al. *Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing*. ICLR 2025. arXiv:2406.08464. https://arxiv.org/abs/2406.08464
13. Singh et al. *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning*. ACL 2024. arXiv:2402.06619. https://arxiv.org/abs/2402.06619
14. Lambert et al. *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124. https://arxiv.org/abs/2411.15124
15. Zhao et al. *WildChat: 1M ChatGPT Interaction Logs in the Wild*. ICLR 2024. arXiv:2405.01470. https://arxiv.org/abs/2405.01470
16. Zheng et al. *LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset*. ICLR 2024. arXiv:2309.11998. https://arxiv.org/abs/2309.11998
17. BAAI. *Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models*. arXiv:2506.11116. https://arxiv.org/abs/2506.11116
18. BAAI. *Scaling Towards the Information Boundary of Instruction Sets: The Infinity Instruct Subject Technical Report*. arXiv:2507.06968. https://arxiv.org/abs/2507.06968
19. Chen et al. *TigerBot: An Open Multilingual Multitask LLM*. arXiv:2312.08688. https://arxiv.org/abs/2312.08688
20. Xu et al. *CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility*. arXiv:2307.09705. https://arxiv.org/abs/2307.09705
21. Liu et al. *AlignBench: Benchmarking Chinese Alignment of Large Language Models*. ACL 2024. arXiv:2311.18743. https://arxiv.org/abs/2311.18743
22. Li et al. *CIF-Bench: A Chinese Instruction-Following Benchmark for Evaluating the Generalizability of Large Language Models*. Findings of ACL 2024. arXiv:2402.13109. https://arxiv.org/abs/2402.13109
23. *CHiSafetyBench: A Chinese Hierarchical Safety Benchmark for Large Language Models*. arXiv:2406.10311. https://arxiv.org/abs/2406.10311
24. Xu et al. *SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark*. arXiv:2307.15020. https://arxiv.org/abs/2307.15020
25. Bai et al. *Qwen Technical Report*. arXiv:2309.16609. https://arxiv.org/abs/2309.16609
26. Yang et al. *Qwen2 Technical Report*. arXiv:2407.10671. https://arxiv.org/abs/2407.10671
27. Qwen Team. *Qwen2.5 Technical Report*. arXiv:2412.15115. https://arxiv.org/abs/2412.15115
28. Qwen Team. *Qwen3 Technical Report*. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
29. Young et al. *Yi: Open Foundation Models by 01.AI*. arXiv:2403.04652. https://arxiv.org/abs/2403.04652
30. Cai et al. *InternLM2 Technical Report*. arXiv:2403.17297. https://arxiv.org/abs/2403.17297
31. DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*. arXiv:2401.02954. https://arxiv.org/abs/2401.02954
32. DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437. https://arxiv.org/abs/2412.19437
33. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948. https://arxiv.org/abs/2501.12948
34. Yang et al. *Baichuan 2: Open Large-scale Language Models*. arXiv:2309.10305. https://arxiv.org/abs/2309.10305
35. GLM Team. *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4*. arXiv:2406.12793. https://arxiv.org/abs/2406.12793
36. Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*. arXiv:2404.06395. https://arxiv.org/abs/2404.06395
37. Wei et al. *Skywork: A More Open Bilingual Foundation Model*. arXiv:2310.19341. https://arxiv.org/abs/2310.19341
38. Skywork. *Skywork-OR1*. arXiv:2505.22312. https://arxiv.org/abs/2505.22312
39. Wang et al. *TeleChat Technical Report*. arXiv:2401.03804. https://arxiv.org/abs/2401.03804
40. Sun et al. *Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters*. arXiv:2411.02265. https://arxiv.org/abs/2411.02265
41. Luo et al. *YAYI 2: Multilingual Open-Source Large Language Models*. arXiv:2312.14862. https://arxiv.org/abs/2312.14862
42. Jiao et al. *Panda LLM: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models*. arXiv:2305.03025. https://arxiv.org/abs/2305.03025
43. Zhang et al. *BayLing: Bridging Cross-lingual Alignment and Instruction Following through Interactive Translation*. arXiv:2306.10968. https://arxiv.org/abs/2306.10968

---

## 15. Access Limitations / 访问限制说明

- **Gated on HuggingFace (`gated: auto`) — no rows readable with the token available in this environment:** `BAAI/Infinity-Instruct`, `lmsys/lmsys-chat-1m`, `BAAI/COIG-PC`, `BAAI/IndustryInstruction`, `BAAI/OL-CC` (401). Their Parquet branches return an auth wall rather than Parquet bytes. Where a redistributed Chinese slice exists (Infinity-Instruct → `Mxode/Chinese-Instruct` config `infinity-instruct`, 386,426 rows; IndustryInstruction → config `industryinstruction`, 358,582 rows), I sampled that instead and said so. | 上述数据集受 HF 门禁限制，本环境令牌无法读取任何行；凡有转发的中文切片，我改为采样该切片并已注明。
- **No auto-converted Parquet:** `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` returns an empty config list, so the viewer/Parquet path yields nothing; it must be fetched as raw files. | 无自动 Parquet 转换，须下载原始文件。
- **Broken conversion job:** `shareAI/ShareGPT-Chinese-English-90k` fails with `ArrowInvalid: JSON parse error: Column(/category) changed from string to array in row 4`. | 转换任务失败。
- **404 to this token:** `deepctrl/deepctrl-sft-data` (figures in this document come from ModelScope / OpenDataLab mirrors and are second-hand), `m-a-p/neo_sft_phase2` (**repo appears removed**), `BAAI/OL-CC`, and `wenge-research/yayi2_sft_data` — **which does not exist at all**; `api/datasets?author=wenge-research` returns only `yayi2_pretrain_data`, `yayi_uie_sft_data`, `yayi_domain_subset` and `TableEval`. The widely-circulated `yayi2_sft_data` id is a propagated citation error and YAYI 2's SFT data was never released. | 本环境令牌返回 404 者如上；其中 `wenge-research/yayi2_sft_data` **根本不存在**，是被反复转引的错误引用。
- **Rate limiting:** `datasets-server.huggingface.co` returned HTTP 429 under sustained querying, which is why the sampling harness was moved to direct Parquet range reads against the CDN. Row counts quoted from `/size` were collected before or between rate-limit windows and are exact. | `datasets-server` 在持续查询下返回 429，故采样改为对 CDN 直接做 Parquet Range 读取；引用的 `/size` 行数取自限流窗口之外，为精确值。
- **Language shares** were computed by reading the full `language` column where one exists, over a stated number of Parquet shards: Aya Dataset (all shards, 202,362 rows), Magpie-Qwen2.5-Pro-1M (2 of 16 shards, 125,000 rows), WildChat-1M (3 of 14 shards, 179,571 rows). Where no `language` field exists, a CJK-density heuristic was used instead and labelled as such: Tulu 3 (2 of 6 shards, 313,115 rows). Percentages are labelled with the sample size they came from; none are extrapolated silently. | 语言占比通过读取完整 `language` 列计算，并注明所覆盖的分片数与行数；所有百分比均标注样本量，未作隐性外推。
- **Not sampled, listed for completeness:** COIG-P, CValues, `liyucheng/zhihu_rlhf_3k`, BAAI/OL-CC, pCLUE, RefGPT, InstructionWild, SmileConv, Alpaca-CoT, Linly, Ziya's described-but-unreleased 5.3M mixture. These appear in the taxonomy and reference list but carry no hands-on quality claim. | 未采样但为完整性列出者如上；它们出现在分类与参考文献中，但不附带任何实测质量判断。
