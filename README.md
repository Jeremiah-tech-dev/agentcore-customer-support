# Customer Support AI Agent — Amazon Bedrock AgentCore + Strands

Udacity AWS AI Engineering — final project. A customer support agent for a
fictional online store, deployed to **AgentCore Runtime**, that tracks
orders, processes refunds, answers product/policy questions, remembers
customers across sessions, calculates loyalty discounts in a sandbox, and
browses live web pages.

All implementation is in [`starter/main.py`](starter/main.py). The original
project instructions are in [`docs/PROJECT_INSTRUCTIONS.md`](docs/PROJECT_INSTRUCTIONS.md).

## Architecture

```
agentcore invoke ──▶ AgentCore Runtime (csagent, Strands Agent, Nova 2 Lite)
                       │
                       ├─ MemoryHook ──────────▶ AgentCore Memory
                       │    before: retrieve facts + preferences, prepend as "Customer Context"
                       │    after:  create_event(user query, assistant reply)
                       │
                       ├─ MCPClient ───────────▶ AgentCore Gateway (MCP, NONE authorizer)
                       │                           ├─ order-tracker    → API Gateway REST (prod) → order-tracker Lambda
                       │                           │     get_order, get_customer, get_customer_orders
                       │                           └─ refund-processor → refund-processor Lambda
                       │                                 initiate_refund, check_refund_status, get_return_label
                       │
                       ├─ search_knowledge_base ─▶ Bedrock Managed Knowledge Base (Retrieve API)
                       ├─ calculate_loyalty_discount ─▶ AgentCore Code Interpreter (fallback: tier-only)
                       └─ AgentCoreBrowser ─────▶ AgentCore Browser
```

## Test results

Full terminal output for every test is in [`test-logs/`](test-logs/).

| # | Scenario | Result | Log |
|---|---|---|---|
| 1 | Order tracking | SHIPPED, UPS, **TRK987654321**, estimated delivery 2026-10-31 | [test1](test-logs/test1-order-tracking.txt) |
| 2 | Refund processing | Refund **REF-OQJVQHHT**, **APPROVED**, credit in **3-5 business days** (+ return label) | [test2](test-logs/test2-refund.txt) |
| 3 | Knowledge Base (RAG) | Platinum: **free same-day shipping, 15% discount, priority support** | [test3](test-logs/test3-knowledge-base.txt) |
| 4 | Cross-session memory | Session `s-A` introduces Jane; new session `s-B` recalls **Jane** + **concise responses** | [4A](test-logs/test4a-memory-session-A.txt), [4B](test-logs/test4b-memory-session-B.txt) |
| 5 | Loyalty discount | 4000 pts redeemed ($40), **10%** Gold discount ($11), final **$99.00**, **250** pts remaining | [test5](test-logs/test5-loyalty-discount.txt) |
| 6 | Browser | Title: *"Learn the Latest Tech Skills; Advance Your Career \| Udacity"* | [test6](test-logs/test6-browser.txt) |

Tests 1 and 2 exercise one API-Gateway-backed and one Lambda-backed Gateway
tool respectively.

Extra verification beyond the agent's replies:
- **Test 4:** the memory resource holds the extracted records *"The user's
  name is Jane."* (namespace `cs_agent/CUST-123/facts`) and *"Prefers concise
  responses"* (`cs_agent/CUST-123/preferences`), so recall comes from
  AgentCore Memory rather than the model.
- **Test 5:** the runtime's CloudWatch logs contain no fallback warning, so
  the numbers came from the Code Interpreter sandbox.

## Implementation notes

- **Memory hook** — retrieval skips tool-result messages and tags each memory
  with its strategy (`[SEMANTIC]`, `[USER_PREFERENCE]`). The save step stores
  the customer's *original* words, not the context-augmented message, so
  retrieved memories are never written back as if the customer said them again.
  Memory failures are logged and never block the customer's answer.
- **Loyalty calculator** — the sandbox code uses `Decimal` with half-up
  rounding for money, and every model-supplied argument is injected with
  `repr()` so it lands in the sandbox as a Python literal, never as code.
  Rules: 100 pts = $1, redeem in blocks of 500, points cover at most 50% of
  the order, tier discount applied after points, earn rates 1/2/5 per $ for
  standard/device/fresh.
- **Gateway tools** are loaded across all pages of `tools/list`.

## Differences from the written instructions

| Instructions say | What was needed | Why |
|---|---|---|
| Gateway targets `order_tracker`, `refund_processor` | `order-tracker`, `refund-processor` | Target names must match `([0-9a-zA-Z][-]?){1,100}`; underscores are rejected |
| Lambda proxy methods as-is | Added a `200` method response to each GET and redeployed `prod` | The Gateway builds tools from the REST API's OpenAPI export, which failed with *"responses is missing"* without one |
| KB with an S3 data source | Managed KB + `MANAGED_KNOWLEDGE_BASE_CONNECTOR` (S3) data source | Managed KBs reject the classic `S3` data-source type. No OpenSearch Serverless collection is created |

## Reflection

**Design decision.** In the `MemoryHook`, I chose to save the customer's
original message to AgentCore Memory rather than the message the model
actually received. `retrieve_customer_context` prepends a "Customer Context"
block of remembered facts to each user message, and the template approach of
walking back to the last user message would have saved that augmented text.
Memory extraction would then re-ingest old memories as if the customer had
just repeated them, duplicating facts and making stale ones look fresh every
turn. The hook keeps the pre-augmentation query and uses it when saving, so
each event contains only what was really said.

**Challenge.** The API Gateway target for `order-tracker` went straight to
`FAILED` with *"paths.'/orders/{order_id}'(get).responses is missing."* The
Lambda worked through the REST API, so the problem was not the function. The
Gateway generates MCP tool definitions from the REST API's exported OpenAPI
document, and Lambda proxy integrations have no method responses by default,
which makes that export invalid. Adding a `200` method response to each GET,
redeploying the `prod` stage and recreating the target fixed it. I then
confirmed all six tools with an MCP client before involving the agent, which
separated Gateway problems from agent problems.

**Production consideration.** The Gateway uses the NONE authorizer, and the
agent trusts whatever `customer_id` the caller sends. In production anyone
who found the URL could call `initiate_refund`, and any caller could claim to
be CUST-123 and read that customer's orders and memories. I would put a
`CUSTOM_JWT` authorizer (for example Amazon Cognito) on both the Runtime and
the Gateway, derive the actor ID from the verified token instead of the
payload, and add a Bedrock Guardrail. For cost and latency I would also
cache the Gateway tool list instead of reconnecting and listing tools on
every request, and alarm on tool error rates through the AgentCore
observability dashboard in CloudWatch.

## Reproduce

```bash
cd starter
uv sync --python 3.13
# Create the resources described in docs/PROJECT_INSTRUCTIONS.md, then fill in
# GATEWAY_URL, KB_ID, REGION, MEMORY_ID in main.py
uv run agentcore configure --entrypoint main.py --name csagent \
  --deployment-type direct_code_deploy --runtime PYTHON_3_13 --disable-memory
uv run agentcore deploy
uv run setup_permissions.py
uv run agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

All AWS resources used for this submission are deleted after grading
(`agentcore destroy`, then Gateway, Memory, Knowledge Base, S3, REST API,
Lambdas and IAM roles).
