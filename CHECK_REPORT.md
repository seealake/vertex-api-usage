# Vertex API 配置检查报告

经过对 `vertex_batch_structured.py` 和 `vertex_batch_grounding.py` 的代码分析，发现以下配置问题和潜在风险：

## 1. `vertex_batch_structured.py` (结构化抽取)

### 🔴 关键问题：Schema 与 Prompt 不匹配
代码中定义了一个全局的 `RESPONSE_SCHEMA` 用于配置 `response_json_schema`，但 `USER_PROMPTS` 列表中包含了两种完全不同的任务：
1. **劳动合同抽取** (姓名, 日期, 薪资, 地点...)
2. **发票信息抽取** (发票号, 金额, 日期, 地址...)

**风险**：Vertex API 的 `response_json_schema` 会强制模型按照指定的 Schema 输出。
- 如果你设置了“合同”的 Schema，那么“发票”的 Prompt 执行时会报错，或者模型会强行编造合同字段来填充发票信息（幻觉）。
- **建议**：
    - 将不同类型的任务拆分到不同的批处理文件中。
    - 或者在代码中支持根据 Prompt 动态选择 Schema（需要修改代码逻辑）。

### ⚠️ 配置缺失：`RESPONSE_SCHEMA` 为空
目前代码中：
```python
RESPONSE_SCHEMA = {}
```
且 `build_config` 中有判断：
```python
if isinstance(RESPONSE_SCHEMA, dict) and RESPONSE_SCHEMA:
    kwargs["response_mime_type"] = "application/json"
    kwargs["response_json_schema"] = RESPONSE_SCHEMA
```
**后果**：由于 Schema 为空，`response_mime_type` 不会被设置为 `application/json`，API 也不会执行结构化约束。模型输出将是普通文本（虽然 Prompt 要求输出 JSON，但没有 API 层面的强制保证），且后续的 `json.loads` 解析逻辑（依赖 `RESPONSE_SCHEMA` 非空判断）可能不会执行或逻辑不一致。

### ℹ️ 系统指令为空
`SYSTEM_INSTRUCTION` 为空字符串。虽然不是错误，但对于结构化抽取任务，明确的系统指令（如“你是一个数据抽取助手...”）通常有助于提高准确率。

## 2. 通用配置问题 (两个文件)

### ❓ `vertexai=True` 与 `api_key` 的组合
代码中使用了：
```python
genai.Client(vertexai=True, api_key=api_key)
```
**需要确认**：
- **标准 Vertex AI (Google Cloud)**：通常使用 `project` 和 `location` 参数，并通过 IAM (ADC) 认证，而不是 `api_key`。
- **Gemini API (AI Studio)**：通常使用 `api_key`，但默认 `vertexai=False`。
- **Vertex AI Express Mode**：如果这是为了使用 Vertex AI 的 Express Mode（允许用 API Key），则是正确的。
- **建议**：请确认你的 API Key 类型。如果是 AI Studio 的 Key，请去掉 `vertexai=True`；如果是 Google Cloud 项目，建议检查是否需要配置 Project ID。

## 3. 代码一致性微瑕
- **System Instruction 格式**：
    - `vertex_batch_structured.py` 传递的是列表 `[SYSTEM_INSTRUCTION]`。
    - `vertex_batch_grounding.py` 传递的是字符串 `SYSTEM_INSTRUCTION`。
    - SDK 通常两者都支持，但保持一致更好。

## 建议后续步骤
1. **拆分任务**：将 `vertex_batch_structured.py` 拆分为 `vertex_batch_contract.py` 和 `vertex_batch_invoice.py`，并分别为它们定义具体的 `RESPONSE_SCHEMA`。
2. **补充 Schema**：根据任务需求填入具体的 JSON Schema。
3. **验证认证方式**：确认 `api_key` 对应的环境，确保 `vertexai=True` 是预期的配置。
