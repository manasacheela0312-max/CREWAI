# CREWAI: 10 CrewAI Projects in One FastAPI Backend

Ten AI agent projects built with **CrewAI Crews and Flows**, served by one **FastAPI** server.
Every agent uses one model: **`gemini-3.1-flash-lite-preview`**. All tools are free.

## API format

Every route is a **POST** request with a JSON body:

```json
{ "input": "your text here" }
```

and every route answers with:

```json
{ "output": "the agent's answer" }
```

| Route | Project | Type |
|---|---|---|
| `POST /agent1` | Blog Post Crew | Crew, sequential |
| `POST /agent2` | Competitor Analysis Crew | Crew, hierarchical |
| `POST /agent3` | Meeting Notes Crew | Crew, parallel tasks |
| `POST /agent4` | Data Report Crew | Crew, structured output |
| `POST /agent5` | Customer Support Flow | Flow, router |
| `POST /agent6` | Resume Screening Flow | Flow, router + crew |
| `POST /agent7` | Bug Triage Flow | Flow, router + 3 crews |
| `POST /agent8` | Content Repurposing Flow | Flow, 3 crews joined with `and_` |
| `POST /agent9` | Lead Qualification Flow | Flow, router + crews |
| `POST /agent10` | Approval Report Flow | Flow, human approval |

Other routes:
- `GET /` lists the routes.
- `GET /health` returns `{"status": "ok"}`.
- `GET /docs` is an interactive page where you can try every route.

## Shared building blocks (in `app.py`)

- `llm`: `LLM(model="gemini/gemini-3.1-flash-lite-preview")`, used by every agent, the manager and direct calls.
- `web_search`: a small custom tool that searches DuckDuckGo with the `ddgs` package. It needs no key.
- `scrape_tool`: `ScrapeWebsiteTool` from `crewai-tools`. It reads a web page's text.
- `make_agent()`: creates an agent with a role, goal and tools.
- `ask()`: one direct LLM call. Flows use it for quick decisions like "billing or technical?".
- `run_crew()`: a one-agent crew. Flows use it as a step.
- `Query` and `Answer`: the shared `{"input"}` request and `{"output"}` response models.

---

## The 10 agents

### 1. Blog Post Crew: `POST /agent1`
**Problem:** Writing a well-researched blog post takes hours of searching, drafting and editing.
**How it works:** Three agents work one after another (`Process.sequential`).
1. **Researcher** searches the web and collects 5 facts with sources.
2. **Writer** writes a 600-word markdown post.
3. **Editor** fixes tone, grammar and unsupported claims.

```json
{ "input": "AI in farming" }
{ "output": "# How AI is changing farming\n..." }
```

### 2. Competitor Analysis Crew: `POST /agent2`
**Problem:** Tracking what competitors ship, charge and say takes hours of browsing every week.
**How it works:** A hierarchical crew (`Process.hierarchical`, `manager_llm`). A manager agent decides who does what among four agents:
- **Site Scraper** reads each website and lists its products.
- **Pricing Analyst** finds each competitor's plans and prices.
- **News Analyst** searches for recent news.
- **Strategist** builds one SWOT table per competitor.

```json
{ "input": "https://www.notion.so, https://clickup.com" }
{ "output": "## Notion\n| Strengths | Weaknesses | ... |" }
```
Competitor names also work, for example `"Notion and ClickUp"`.

### 3. Meeting Notes Crew: `POST /agent3`
**Problem:** After every meeting someone must write the summary and chase action items.
**How it works:**
- **Summarizer**, **Action Finder** and **Decision Logger** run at the same time (`async_execution=True`).
- A **Formatter** waits for all three and combines them into one document.

```json
{ "input": "Asha: Let's ship v2 on Friday. Ravi: I'll fix the login bug by Wednesday." }
{ "output": "## Summary\n...\n## Action Items\n- Ravi - fix login bug - Wednesday\n## Decisions\n- Ship v2 on Friday" }
```

### 4. Data Report Crew: `POST /agent4`
**Problem:** Turning sales CSV files into a readable report is slow, repetitive work.
**How it works:**
1. **pandas** reads the CSV and computes the statistics, so the maths is exact.
2. **Data Analyst** pulls out structured headline numbers and trends (`output_pydantic`).
3. **Insight Writer** writes the top 3 findings.
4. **Reviewer** checks every number against the data.

```json
{ "input": "date,region,sales\n2026-10-01,North,1200\n2026-10-02,South,950" }
{ "output": "## Top 3 findings\n1. North sold 26% more than South ..." }
```
If the input is not CSV with a header row and data, the output explains the expected format.

### 5. Customer Support Flow: `POST /agent5`
**Problem:** Support tickets arrive mixed together, and each must be read, sorted and sent to the right person.
**How it works:**
1. `receive_ticket` (`@start`) loads the ticket.
2. `categorize_ticket` makes an LLM call that picks billing, account_access, technical or feature_request.
3. `route_by_category` (`@router`) sends the ticket to `handle_billing`, `handle_access` or `handle_other`. Billing and access tickets get high priority.
4. `resolve_ticket` (`@listen(or_(...))`) writes the reply.

```json
{ "input": "I was charged twice for my plan this month." }
{ "output": "Category: billing\nPriority: high\n\nSorry about that! ..." }
```

### 6. Resume Screening Flow: `POST /agent6`
**Problem:** One job post brings hundreds of resumes, and reading each takes recruiters days.
**How it works:**
1. `score_resume` makes an LLM call that scores the fit from 0 to 10.
2. `check_score` (`@router`) sends scores of 7 or more to the **Summary Crew** (strengths and gaps). Lower scores go to `draft_rejection` (a kind email).
3. `save_results` (`or_`) finishes the flow.

Send the job description and the resume together in one text:
```json
{ "input": "JOB: Python backend developer, 3+ years, FastAPI\nRESUME: Priya - 4 years Python, FastAPI, AWS" }
{ "output": "Score: 8/10\nDecision: shortlisted\n\n## Strengths\n..." }
```

### 7. Bug Triage Flow: `POST /agent7`
**Problem:** Issues pile up without labels, so urgent bugs sit next to feature requests.
**How it works:**
1. `classify_issue` decides between bug, feature and question.
2. `route_type` (`@router`) sends the issue to one of three crews:
   - **Bug Crew:** QA Engineer gives severity, likely cause and steps.
   - **Feature Crew:** Product Manager writes a user story with acceptance criteria.
   - **Answer Crew:** Support Engineer answers, searching the web if needed.
3. `post_reply` returns the labels and the reply.

```json
{ "input": "App crashes when I upload a PNG. Steps: open editor, upload a 5MB PNG." }
{ "output": "Type: bug\nLabels: bug, severity:high\n\n**Severity:** high ..." }
```

### 8. Content Repurposing Flow: `POST /agent8`
**Problem:** One article should become a LinkedIn post, a tweet thread and a newsletter, but rewriting it three times wastes a day.
**How it works:**
1. `load_article` takes the text, or scrapes the page if the input is a URL.
2. `extract_points` pulls out 5 key points.
3. **LinkedIn Crew**, **Twitter Crew** and **Newsletter Crew** all listen to `extract_points` and run side by side.
4. `publish_pack` waits for all three with `@listen(and_(...))`.

```json
{ "input": "https://example.com/my-article" }
{ "output": "## LinkedIn post\n...\n\n## Twitter thread\n1/6 ...\n\n## Newsletter\n..." }
```

### 9. Lead Qualification Flow: `POST /agent9`
**Problem:** Sales teams waste time on cold leads while hot leads wait days for a reply.
**How it works:**
1. **Enrich Crew** reads the lead's website (or searches the web) and writes a company profile.
2. `score_lead` (`@router`) compares the profile with the ideal customer ("B2B software companies with 50+ employees") and returns hot, warm or cold:
   - **hot:** the **Outreach Crew** drafts a personal email.
   - **warm:** the lead is added to the nurture list.
   - **cold:** the lead is archived.
3. `log_lead` (`or_`) returns the result.

```json
{ "input": "Rahul Mehta, https://www.zoho.com" }
{ "output": "Tier: hot\nAction: Send a personal email today\n\n## Company\n...\n\n## Email draft\nSubject: ..." }
```

### 10. Approval Report Flow: `POST /agent10`
**Problem:** Auto-generated reports are risky to publish unchecked, but reviewing every word by hand is slow.
**How it works:** A person approves each report before it is final. `check_decision` (`@router`) picks the path:
- **new:** the **Research Crew** collects facts with web search, then the **Writing Crew** drafts the report.
- **approved:** `publish_report` marks it published.
- **revise:** the **Revision Crew** rewrites the draft using the feedback, then waits for review again.

Because every route takes plain text, the person's decision is sent as text too:

```json
// step 1: send a topic
{ "input": "Sources of protein for vegetarians" }
{ "output": "# Draft ...\n---\nReport ID: a1b2c3d4 (status: awaiting review)\nTo publish it, send: approve a1b2c3d4\nTo change it, send: revise a1b2c3d4: what to change" }

// step 2: ask for changes (repeat as often as you like)
{ "input": "revise a1b2c3d4: add protein grams per serving" }

// step 3: approve
{ "input": "approve a1b2c3d4" }
{ "output": "Status: published\n\n# Final report ..." }
```

Drafts are kept in memory, so they are cleared when the server restarts.

---

## Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a file named `.env` next to `app.py` with your Gemini key (free at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key
```

Start the server:

```bash
uvicorn app:app
```

The API runs at http://localhost:8000. Each request can take 20 to 90 seconds, because several agents call the model one after another.

**Never commit your `.env` file.** Only `app.py`, `requirements.txt` and `README.md` belong in this repo.

## Test in Postman

1. Set the method to **POST** and the URL to `http://localhost:8000/agent1` (or your deployed URL).
2. Open **Body**, choose **raw**, then **JSON**, and enter `{"input": "AI in farming"}`.
3. Click **Send**. The response is `{"output": "..."}`.
4. Repeat for `/agent2` to `/agent10` using the example inputs above.

Postman's default timeout is fine, but the first request on a sleeping Render server can take an extra minute.

