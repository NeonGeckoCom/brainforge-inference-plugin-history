# BrainForge Inference Plugin: History

Request step that validates and truncates `LLMRequest.history` after earlier
request steps have prepended context. It repairs history rather than skipping:
unknown messages are dropped, prior citation markup is stripped, and
conversational turns are truncated to complete user/llm pairs.

Entry point group: `brainforge.inference.request_steps`

| Step id | Class |
| --- | --- |
| `brainforge.history` | `RequestHistoryStep` |

```yaml
pipeline:
  request:
    - id: brainforge.history
```

Run this step last on the request pipeline. It writes
`transaction.request_step_context["brainforge.history"]`.

### Fitting conversation to a model window

Set the history step's `config: {max_tokens: 8192}` for a model with an
8192-token context window. Choose the limit for the deployed model. By default
the window is unset and the original pair-count policy applies.

The step estimates each message as four framing tokens plus one token per four
UTF-8 bytes (rounded up). It reserves the larger of the request's output cap
(`request.max_tokens`, normally 512) and `extra_body.max_new_tokens`, then
accounts for the query, supplied system prompt and all cleaned information,
instruction and tool messages. Remaining space holds the longest sequence of
recent complete pairs that fits. An explicitly supplied pair cap still applies
(including one serialized by a client). With no explicit cap, the window can
retain more than the request model's implicit two pairs.

Mandatory context is preserved even when it alone plus the output reservation
exceeds the estimate. In that case the step keeps zero pairs and reports
`context_budget_exceeded` in its step context. Actual counts depend on the
model tokenizer, chat template and any instructions added later by the host.

### How deep can history remain useful?

The plugin can pass any number of fitting pairs. Retention alone does not
measure whether a model uses a fact. [Lost in the Middle](https://arxiv.org/abs/2307.03172)
found that recall can depend on the position of relevant information, with
worse performance in the middle on the evaluated tasks. That evidence does
not establish a universal turn cutoff. To evaluate a deployment, place a
random code in one of several equal-length exchanges, move it between oldest,
middle and newest positions, and ask for it in the final query. Verify the code
survived truncation, then compare exact-match accuracy over multiple codes at
each depth, holding the model, total input length and output cap constant.
