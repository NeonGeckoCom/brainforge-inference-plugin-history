# Neon Inference Plugin: History

Request step that validates and truncates `LLMRequest.history` after earlier
request steps have prepended context. It repairs history rather than skipping:
unknown messages are dropped, prior citation markup is stripped, and
conversational turns are truncated to complete user/llm pairs.

Entry point group: `brainforge.inference.request_steps`

| Step id | Class |
| --- | --- |
| `neon.history` | `RequestHistoryStep` |

```yaml
pipeline:
  request:
    - id: neon.history
```

Run this step last on the request pipeline. It writes
`transaction.request_step_context["neon.history"]`.
