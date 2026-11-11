# ai-support-bot

一个简单、可回放的客服 agent：固定的多段式工作流、只读工具、带时间轴的知识库。

[English](README.md)

全部代码约 1400 行 Python，只依赖 `requests`，不用任何 agent 框架：流程是普通代码，
模型在每一段只负责给出结构化输出。

## 设计要点

- **流程不交给模型决定。** 分诊、调查、质检、安全收口是各自独立的调用，各有自己的 schema，段与段之间由代码控制。
- **先核实再回答。** 分诊给每条诉求打上类别；工具声明自己对哪些类别是必查的，没查完 `submit_reply` 会被驳回。
- **工具不写数据。** 需要改数据的工具只登记一个待执行动作：回复真正发出时才执行，回复里没有引用它就作废，
  执行前还会按实时数据复核。
- **每条回复都有第二个人看。** 独立的质检调用对照知识库和工具证据挑错，critical 问题最多驱动两轮修订。
- **退款和补偿单独设防。** 可以复述已公布的规则；承诺和资格判断会被识别并删掉，工单转人工。
- **可回放。** 每条知识带 `available_at` / `valid_to`，每个工具都拿到工单时间，回放旧工单时只能看到当时可知的信息，
  所以可以拿历史人工回复做离线评测。

## 工作流

```
工单 ─► 分诊 ─► 检索 (BM25) ─► 工具循环 ─► submit_reply ─► 质检 ─► 安全收口 ─► 发送 / 草稿
                                  ▲                          │
                                  └──── 修订（最多 2 轮）◄────┘
```

1. **分诊**：拆诉求、归类（这一步不带知识库，调用很小）。
2. **检索**：按整单和每条诉求各做一次 BM25，结果贴在作答前（全量知识库也在上下文里，放在缓存断点之后）。
3. **作答**：工具循环，必须以 `submit_reply` 结束，决定四选一：`answer`、`answer_and_handoff`、`handoff`、`clarify`。
4. **质检**：问题分 critical / minor，只有 critical 触发修订。
5. **安全收口**：对剩余问题做窄口径处理——转人工、删掉退款承诺、压缩过长回复、丢弃回复没引用的动作；
   其他仍为 critical 的回复不自动发送，留作草稿。

## 快速开始

```bash
pip install -e .
cp .env.example .env          # 填 SB_ANTHROPIC_API_KEY（或 SB_OPENAI_* 一对）
supportbot init-demo          # 生成 demo.sqlite：两个用户、两张未回复工单
supportbot ask examples/ticket.json --trace   # 跑单个案例，打印完整轨迹
supportbot serve --once       # 处理演示工单库里所有待回复工单
supportbot drafts             # 查看待审批草稿（SB_AUTO_SEND=0 时）
supportbot approve 1
```

测试用脚本化的假模型离线运行：

```bash
python -m unittest discover -s tests
```

## 配置

| 变量 | 默认值 | 含义 |
|---|---|---|
| `SB_BRAND` | `Acme` | 提示词里的产品名 |
| `SB_MODEL` | `claude-haiku-5-5` | `claude-*` 走 Anthropic Messages API，其余走 OpenAI 兼容的 Responses API |
| `SB_EFFORT` | `medium` | 作答和质检的思考强度（`low` … `max`） |
| `SB_ANTHROPIC_API_KEY`、`SB_ANTHROPIC_BASE_URL` | | Anthropic 后端 |
| `SB_OPENAI_API_KEY`、`SB_OPENAI_BASE_URL` | | Responses 后端 |
| `SB_THINKING` | `adaptive` | `off` 不发送思考参数（给不支持的模型用） |
| `SB_KB_FILE` | `examples/kb.jsonl` | 知识库 |
| `SB_POLICY_FILE` | | 追加到内置规则后面的业务规则 |
| `SB_TOOLS` | `examples/demo_tools.py` | 注册工具的 Python 文件，逗号分隔 |
| `SB_TZ_OFFSET_HOURS` | `0` | 用户本地时间相对 UTC 的偏移（北京时间填 8） |
| `SB_MAX_REPLY_CHARS` | `600` | 超过就压缩一次；`0` 关闭 |
| `SB_AUTO_SEND` | `0` | `1` 表示回复不经审批直接发出 |
| `SB_DB` | `demo.sqlite` | 演示工单库 |

## 知识库

每行一个 JSON 对象，字段见 [README.md](README.md#knowledge-base)：

- `available_at`（UTC）是客服最早能知道这条事实的时间：之后的工单能看到，回放更早的工单看不到。
- `valid_to`（本地日期）让事实失效。最近 14 天内 `incident` / `promo` / `status` 类条目会置顶为“近期临时状态”，
  优先于常规规则。

## 写工具

示例见 `examples/demo_tools.py`：用 `@tool` 注册只读工具，`required_for` 声明必查类别；
要改数据就 `ctx.propose(...)` 登记动作并用 `@executor` 提供执行函数，执行函数里要按实时数据再复核一次。
类别列表在 `supportbot/prompts.py` 的 `CATEGORIES`。接入自己的工单系统只需替换 `supportbot/store.py`：
产出案例 dict，并负责把最终回复发出去。

## 开自动发送之前

先用 `SB_AUTO_SEND=0` 跑一段时间，读草稿和轨迹；用 `--kb-freeze` 回放几周真实工单，和人工回复对比。
写操作要小、有上限，并在执行函数里复核。

## 许可证

MIT
