"""
CrewAI Projects - 10 agents behind one FastAPI server.

Every route is a POST that takes   {"input": "some text"}
and returns                        {"output": "some text"}

/agent1 .. /agent4   -> Crews  (teams of agents)
/agent5 .. /agent10  -> Flows  (steps, routers and crews joined together)

Every agent uses one model: gemini-3.1-flash-lite-preview
Set GEMINI_API_KEY in a .env file (local) or in your hosting dashboard.
"""

import io
import os
import re
import traceback
import uuid
from pathlib import Path

import pandas as pd
from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")  # always read the .env next to app.py
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    print("\n*** GEMINI_API_KEY is not set. Add it to a .env file next to app.py "
          "(GEMINI_API_KEY=your-key) or to your hosting dashboard. ***\n")

# ---------------------------------------------------------------------------
# Shared setup: one model, two free tools and a few small helpers
# ---------------------------------------------------------------------------

llm = LLM(model="gemini/gemini-3.1-flash-lite-preview", temperature=0.3)
scrape_tool = ScrapeWebsiteTool()


@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top 5 results."""
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


def make_agent(role: str, goal: str, tools: list | None = None) -> Agent:
    return Agent(role=role, goal=goal, backstory=f"You are an expert {role}.",
                 llm=llm, tools=tools or [], verbose=False)


def ask(prompt: str) -> str:
    """A direct LLM call, used by Flows for quick decisions."""
    return str(llm.call(prompt)).strip()


def run_crew(role: str, goal: str, description: str, expected: str,
             inputs: dict, tools: list | None = None) -> str:
    """A one-agent crew, used as a step inside Flows."""
    agent = make_agent(role, goal, tools)
    task = Task(description=description, expected_output=expected, agent=agent)
    return Crew(agents=[agent], tasks=[task]).kickoff(inputs=inputs).raw


def pick(answer: str, options: list[str], default: str) -> str:
    """Turn a free-text LLM answer into one of the allowed labels."""
    answer = answer.lower()
    return next((o for o in options if o in answer), default)


# Every route uses the same request and response shape
class Query(BaseModel):
    input: str = Field(min_length=1, examples=["AI in farming"])


class Answer(BaseModel):
    output: str


app = FastAPI(title="CrewAI Projects", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
def show_error(request: Request, exc: Exception):
    """Send the real error back in the same {"output": ...} shape."""
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"output": f"Error: {type(exc).__name__}: {exc}"[:500]})


@app.get("/")
def home():
    return {"message": "CrewAI agents API. POST {\"input\": \"...\"} to any route.",
            "routes": [f"/agent{i}" for i in range(1, 11)], "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Agent 1 - Blog Post Crew (sequential: Researcher -> Writer -> Editor)
# input: a topic
# ---------------------------------------------------------------------------

@app.post("/agent1", response_model=Answer, tags=["Crews"])
def agent1(q: Query):
    researcher = make_agent("Researcher", "Find accurate, recent facts on {topic}", [web_search])
    writer = make_agent("Writer", "Write clear, engaging blog posts")
    editor = make_agent("Editor", "Polish posts for tone, flow and accuracy")

    research = Task(description="Research {topic}. Collect 5 key facts with their sources.",
                    expected_output="5 bullet points of facts with source links", agent=researcher)
    write = Task(description="Write a 600-word blog post on {topic} using the research.",
                 expected_output="A markdown blog post with a title and subheadings", agent=writer)
    edit = Task(description="Edit the draft: fix tone, grammar and any unsupported claims.",
                expected_output="The final blog post in markdown", agent=editor)

    crew = Crew(agents=[researcher, writer, editor], tasks=[research, write, edit],
                process=Process.sequential)
    return Answer(output=crew.kickoff(inputs={"topic": q.input}).raw)


# ---------------------------------------------------------------------------
# Agent 2 - Competitor Analysis Crew (hierarchical: a manager delegates)
# input: competitor websites or names, e.g. "https://www.notion.so, https://clickup.com"
# ---------------------------------------------------------------------------

@app.post("/agent2", response_model=Answer, tags=["Crews"])
def agent2(q: Query):
    agents = [
        make_agent("Site Scraper", "Read competitor websites and list their products", [scrape_tool]),
        make_agent("Pricing Analyst", "Find each competitor's plans and prices", [scrape_tool, web_search]),
        make_agent("News Analyst", "Find recent news about each competitor", [web_search]),
        make_agent("Strategist", "Turn findings into a clear SWOT analysis"),
    ]
    research = Task(description="Study these competitors: {competitors}. Find their products, pricing and recent news.",
                    expected_output="Notes on products, pricing and news for each competitor")
    swot = Task(description="Write a competitor report with a SWOT table for each competitor.",
                expected_output="A markdown report with one SWOT table per competitor")

    crew = Crew(agents=agents, tasks=[research, swot],
                process=Process.hierarchical, manager_llm=llm)
    return Answer(output=crew.kickoff(inputs={"competitors": q.input}).raw)


# ---------------------------------------------------------------------------
# Agent 3 - Meeting Notes Crew (3 tasks run in parallel, then a Formatter)
# input: a meeting transcript
# ---------------------------------------------------------------------------

@app.post("/agent3", response_model=Answer, tags=["Crews"])
def agent3(q: Query):
    summarizer = make_agent("Summarizer", "Summarize meetings in 5 lines")
    finder = make_agent("Action Finder", "Find every action item with owner and due date")
    logger = make_agent("Decision Logger", "List every decision made")
    formatter = make_agent("Formatter", "Combine notes into a clean structure")

    t1 = Task(description="Summarize this meeting in 5 lines:\n{transcript}",
              expected_output="A 5-line summary", agent=summarizer, async_execution=True)
    t2 = Task(description="List action items as 'Owner - task - due date':\n{transcript}",
              expected_output="A list of action items", agent=finder, async_execution=True)
    t3 = Task(description="List the decisions made:\n{transcript}",
              expected_output="A list of decisions", agent=logger, async_execution=True)
    final = Task(description="Combine the summary, action items and decisions into one document.",
                 expected_output="Markdown with three sections: Summary, Action Items, Decisions",
                 agent=formatter, context=[t1, t2, t3])

    crew = Crew(agents=[summarizer, finder, logger, formatter], tasks=[t1, t2, t3, final])
    return Answer(output=crew.kickoff(inputs={"transcript": q.input}).raw)


# ---------------------------------------------------------------------------
# Agent 4 - Data Report Crew (pandas stats -> structured stats -> insights -> review)
# input: CSV text with a header row
# ---------------------------------------------------------------------------

class SalesStats(BaseModel):
    headline_numbers: list[str]
    trends: list[str]


@app.post("/agent4", response_model=Answer, tags=["Crews"])
def agent4(q: Query):
    try:
        df = pd.read_csv(io.StringIO(q.input))
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return Answer(output="Please send CSV text with a header row and at least one data row, e.g.\n"
                             "date,region,sales\n2026-10-01,North,1200\n2026-10-02,South,950")
    summary = df.describe(include="all").to_string()  # pandas does the maths

    analyst = make_agent("Data Analyst", "Read numbers and spot the key figures")
    writer = make_agent("Insight Writer", "Explain data in plain words")
    reviewer = make_agent("Reviewer", "Check every number against the data")

    stats = Task(description="Here is a pandas summary of the data ({rows} rows):\n{summary}\n\n"
                             "First rows:\n{head}\n\nPull out the headline numbers and trends.",
                 expected_output="Headline numbers and trends", agent=analyst, output_pydantic=SalesStats)
    insights = Task(description="Write a short report with the top 3 findings from the stats.",
                    expected_output="A markdown report with 3 findings", agent=writer)
    review = Task(description="Check the report against the stats and fix any wrong number.",
                  expected_output="The final markdown report", agent=reviewer)

    crew = Crew(agents=[analyst, writer, reviewer], tasks=[stats, insights, review])
    result = crew.kickoff(inputs={"summary": summary, "head": df.head(10).to_string(), "rows": len(df)})
    return Answer(output=result.raw)


# ---------------------------------------------------------------------------
# Agent 5 - Customer Support Flow (router picks one handler)
# input: a customer ticket
# ---------------------------------------------------------------------------

class SupportState(BaseModel):
    ticket: str = ""
    category: str = ""
    priority: str = "normal"
    resolution: str = ""


class SupportFlow(Flow[SupportState]):
    @start()
    def receive_ticket(self):
        self.state.ticket = self.state.ticket.strip()

    @listen(receive_ticket)
    def categorize_ticket(self):
        answer = ask("Classify this support ticket as exactly one word: billing, "
                     f"account_access, technical or feature_request.\n\nTicket: {self.state.ticket}")
        self.state.category = pick(answer, ["billing", "account_access", "technical", "feature_request"], "technical")

    @router(categorize_ticket)
    def route_by_category(self):
        return self.state.category if self.state.category in ("billing", "account_access") else "other"

    @listen("billing")
    def handle_billing(self):
        self.state.priority = "high"

    @listen("account_access")
    def handle_access(self):
        self.state.priority = "high"

    @listen("other")
    def handle_other(self):
        self.state.priority = "normal"

    @listen(or_(handle_billing, handle_access, handle_other))
    def resolve_ticket(self):
        self.state.resolution = ask(f"You are a friendly support agent. Write a short reply that solves this "
                                    f"{self.state.category} ticket:\n{self.state.ticket}")


@app.post("/agent5", response_model=Answer, tags=["Flows"])
def agent5(q: Query):
    flow = SupportFlow()
    flow.kickoff(inputs={"ticket": q.input})
    s = flow.state
    return Answer(output=f"Category: {s.category}\nPriority: {s.priority}\n\n{s.resolution}")


# ---------------------------------------------------------------------------
# Agent 6 - Resume Screening Flow (score 7+ -> shortlist, else reject)
# input: the job description and the resume, in one text
# ---------------------------------------------------------------------------

class ResumeState(BaseModel):
    text: str = ""
    score: int = 0
    decision: str = ""
    message: str = ""


class ResumeFlow(Flow[ResumeState]):
    @start()
    def load_resume(self):
        self.state.text = self.state.text.strip()

    @listen(load_resume)
    def score_resume(self):
        answer = ask("The text below has a job description and a candidate's resume. Score how well the "
                     "candidate fits the job from 0 to 10. If there is no job description, judge the resume "
                     f"on its own. Reply with only the number.\n\n{self.state.text}")
        digits = "".join(c for c in answer if c.isdigit())[:2]
        self.state.score = min(int(digits or 0), 10)

    @router(score_resume)
    def check_score(self):
        return "shortlist" if self.state.score >= 7 else "reject"

    @listen("shortlist")
    def summary_crew(self):
        self.state.decision = "shortlisted"
        self.state.message = run_crew(
            "Recruiter", "Summarize candidates for hiring managers",
            "The text has a job description and a resume. List the candidate's strengths and gaps.\n\n{text}",
            "A markdown summary with Strengths and Gaps sections", {"text": self.state.text})

    @listen("reject")
    def draft_rejection(self):
        self.state.decision = "rejected"
        self.state.message = ask("Write a short, kind rejection email for the candidate in this text. "
                                 f"Do not list their weaknesses.\n\n{self.state.text}")

    @listen(or_(summary_crew, draft_rejection))
    def save_results(self):
        return self.state


@app.post("/agent6", response_model=Answer, tags=["Flows"])
def agent6(q: Query):
    flow = ResumeFlow()
    flow.kickoff(inputs={"text": q.input})
    s = flow.state
    return Answer(output=f"Score: {s.score}/10\nDecision: {s.decision}\n\n{s.message}")


# ---------------------------------------------------------------------------
# Agent 7 - Bug Triage Flow (bug / feature / question -> its own crew)
# input: a GitHub issue (title and details)
# ---------------------------------------------------------------------------

class TriageState(BaseModel):
    issue: str = ""
    issue_type: str = ""
    labels: list[str] = []
    reply: str = ""


class TriageFlow(Flow[TriageState]):
    @start()
    def read_issue(self):
        self.state.issue = self.state.issue.strip()

    @listen(read_issue)
    def classify_issue(self):
        answer = ask(f"Is this GitHub issue a bug, feature or question? Reply with one word.\n\n{self.state.issue}")
        self.state.issue_type = pick(answer, ["bug", "feature", "question"], "question")

    @router(classify_issue)
    def route_type(self):
        return self.state.issue_type

    @listen("bug")
    def bug_crew(self):
        severity = pick(ask(f"Rate this bug's severity as low, medium or high. One word.\n{self.state.issue}"),
                        ["high", "medium", "low"], "medium")
        self.state.labels = ["bug", f"severity:{severity}"]
        self.state.reply = run_crew("QA Engineer", "Triage bugs clearly",
                                    "Triage this bug. Give severity, likely cause and steps to reproduce.\n{issue}",
                                    "A short markdown triage note", {"issue": self.state.issue})

    @listen("feature")
    def feature_crew(self):
        self.state.labels = ["enhancement"]
        self.state.reply = run_crew("Product Manager", "Turn requests into user stories",
                                    "Write a user story with acceptance criteria for this request.\n{issue}",
                                    "A markdown user story with acceptance criteria", {"issue": self.state.issue})

    @listen("question")
    def answer_crew(self):
        self.state.labels = ["question"]
        self.state.reply = run_crew("Support Engineer", "Answer user questions helpfully",
                                    "Answer this question. Search the web if needed.\n{issue}",
                                    "A helpful markdown answer", {"issue": self.state.issue}, [web_search])

    @listen(or_(bug_crew, feature_crew, answer_crew))
    def post_reply(self):
        return self.state


@app.post("/agent7", response_model=Answer, tags=["Flows"])
def agent7(q: Query):
    flow = TriageFlow()
    flow.kickoff(inputs={"issue": q.input})
    s = flow.state
    return Answer(output=f"Type: {s.issue_type}\nLabels: {', '.join(s.labels)}\n\n{s.reply}")


# ---------------------------------------------------------------------------
# Agent 8 - Content Repurposing Flow (3 crews side by side, joined with and_)
# input: article text, or a URL to an article
# ---------------------------------------------------------------------------

class RepurposeState(BaseModel):
    article: str = ""
    key_points: str = ""
    linkedin: str = ""
    twitter_thread: str = ""
    newsletter: str = ""


class RepurposeFlow(Flow[RepurposeState]):
    @start()
    def load_article(self):
        text = self.state.article.strip()
        if text.startswith("http") and " " not in text:
            self.state.article = scrape_tool.run(website_url=text)[:15000]

    @listen(load_article)
    def extract_points(self):
        self.state.key_points = ask(f"List the 5 key points of this article:\n\n{self.state.article}")

    @listen(extract_points)
    def linkedin_crew(self):
        self.state.linkedin = run_crew("LinkedIn Writer", "Write engaging LinkedIn posts",
                                       "Write a 150-word LinkedIn post with a strong hook from:\n{points}",
                                       "A LinkedIn post", {"points": self.state.key_points})

    @listen(extract_points)
    def twitter_crew(self):
        self.state.twitter_thread = run_crew("Twitter Writer", "Write punchy threads",
                                             "Write a 6-tweet thread, numbered 1/6 to 6/6, from:\n{points}",
                                             "A 6-tweet thread", {"points": self.state.key_points})

    @listen(extract_points)
    def newsletter_crew(self):
        self.state.newsletter = run_crew("Newsletter Writer", "Write friendly newsletters",
                                         "Write a newsletter section: a short intro and 3 bullets, from:\n{points}",
                                         "A newsletter section in markdown", {"points": self.state.key_points})

    @listen(and_(linkedin_crew, twitter_crew, newsletter_crew))
    def publish_pack(self):
        return self.state


@app.post("/agent8", response_model=Answer, tags=["Flows"])
def agent8(q: Query):
    flow = RepurposeFlow()
    flow.kickoff(inputs={"article": q.input})
    s = flow.state
    return Answer(output=f"## LinkedIn post\n\n{s.linkedin}\n\n## Twitter thread\n\n{s.twitter_thread}"
                         f"\n\n## Newsletter\n\n{s.newsletter}")


# ---------------------------------------------------------------------------
# Agent 9 - Lead Qualification Flow (enrich -> hot / warm / cold)
# input: the lead's name and company website, e.g. "Rahul Mehta, https://www.zoho.com"
# ---------------------------------------------------------------------------

IDEAL_CUSTOMER = "B2B software companies with 50+ employees"


class LeadState(BaseModel):
    lead: str = ""
    company_summary: str = ""
    tier: str = ""
    action: str = ""
    email_draft: str = ""


class LeadFlow(Flow[LeadState]):
    @start()
    def capture_lead(self):
        self.state.lead = self.state.lead.strip()

    @listen(capture_lead)
    def enrich_crew(self):
        self.state.company_summary = run_crew(
            "Lead Researcher", "Understand what a company does",
            "Research this sales lead: {lead}. Read their website if one is given, otherwise search the web. "
            "Describe the company: what it sells, who to, and its size.",
            "A 5-line company profile", {"lead": self.state.lead}, [scrape_tool, web_search])

    @router(enrich_crew)
    def score_lead(self):
        answer = ask(f"Our ideal customer: {IDEAL_CUSTOMER}\n\nCompany:\n{self.state.company_summary}\n\n"
                     "How good a fit is this lead? Reply with one word: hot, warm or cold.")
        self.state.tier = pick(answer, ["hot", "warm", "cold"], "warm")
        return self.state.tier

    @listen("hot")
    def outreach_crew(self):
        self.state.action = "Send a personal email today"
        self.state.email_draft = run_crew(
            "Sales Rep", "Write short personal outreach emails",
            "Write a 120-word email to the person in this lead: {lead}\n\nAbout their company:\n{summary}",
            "A sales email with a subject line",
            {"lead": self.state.lead, "summary": self.state.company_summary})

    @listen("warm")
    def add_to_nurture(self):
        self.state.action = "Add to the weekly tips newsletter"

    @listen("cold")
    def archive_lead(self):
        self.state.action = "Archive, no action now"

    @listen(or_(outreach_crew, add_to_nurture, archive_lead))
    def log_lead(self):
        return self.state


@app.post("/agent9", response_model=Answer, tags=["Flows"])
def agent9(q: Query):
    flow = LeadFlow()
    flow.kickoff(inputs={"lead": q.input})
    s = flow.state
    out = f"Tier: {s.tier}\nAction: {s.action}\n\n## Company\n\n{s.company_summary}"
    if s.email_draft:
        out += f"\n\n## Email draft\n\n{s.email_draft}"
    return Answer(output=out)


# ---------------------------------------------------------------------------
# Agent 10 - Approval Report Flow (a person approves before anything is final)
# input, step 1: a topic                        -> a draft and a report ID
# input, step 2: "approve <id>"                 -> the final report
#            or: "revise <id>: what to change"  -> a new draft to review again
# ---------------------------------------------------------------------------

DRAFTS: dict[str, dict] = {}  # in-memory store; drafts reset when the server restarts
REVIEW_CMD = re.compile(r"^\s*(approve|revise)\s+([0-9a-f]{8})\b\s*:?\s*(.*)$", re.I | re.S)


class ApprovalState(BaseModel):
    topic: str = ""
    draft: str = ""
    decision: str = ""
    feedback: str = ""
    status: str = ""


class ApprovalFlow(Flow[ApprovalState]):
    @start()
    def collect_topic(self):
        pass

    @router(collect_topic)
    def check_decision(self):
        return self.state.decision or "new"

    @listen("new")
    def research_and_write(self):
        notes = run_crew("Researcher", "Find accurate facts",
                         "Research {topic}. Collect key facts with sources.",
                         "Bullet points of facts with source links", {"topic": self.state.topic}, [web_search])
        self.state.draft = run_crew("Report Writer", "Write clear reports",
                                    "Write a report on {topic} using these notes:\n{notes}",
                                    "A markdown report with a title, sections and sources",
                                    {"topic": self.state.topic, "notes": notes})
        self.state.status = "awaiting_review"

    @listen("approved")
    def publish_report(self):
        self.state.status = "published"

    @listen("revise")
    def revision_crew(self):
        self.state.draft = run_crew("Editor", "Revise reports based on feedback",
                                    "Revise this report using the feedback.\n\nFeedback: {feedback}\n\nReport:\n{draft}",
                                    "The revised markdown report",
                                    {"feedback": self.state.feedback, "draft": self.state.draft})
        self.state.status = "awaiting_review"


@app.post("/agent10", response_model=Answer, tags=["Flows"])
def agent10(q: Query):
    cmd = REVIEW_CMD.match(q.input)
    if cmd:  # step 2: a person reviews an existing draft
        decision, report_id, feedback = cmd.group(1).lower(), cmd.group(2).lower(), cmd.group(3).strip()
        if report_id not in DRAFTS:
            return Answer(output=f"No draft found with ID {report_id}. Send a topic to start a new report.")
        if decision == "revise" and not feedback:
            return Answer(output=f"Say what to change, e.g. revise {report_id}: add more examples")
        state = {**DRAFTS[report_id], "decision": "approved" if decision == "approve" else "revise",
                 "feedback": feedback}
    else:  # step 1: a new topic
        report_id = uuid.uuid4().hex[:8]
        state = {"topic": q.input.strip()}

    flow = ApprovalFlow()
    flow.kickoff(inputs=state)
    s = flow.state
    if s.status == "published":
        DRAFTS.pop(report_id, None)
        return Answer(output=f"Status: published\n\n{s.draft}")

    DRAFTS[report_id] = {"topic": s.topic, "draft": s.draft}
    return Answer(output=f"{s.draft}\n\n---\nReport ID: {report_id} (status: awaiting review)\n"
                         f"To publish it, send: approve {report_id}\n"
                         f"To change it, send: revise {report_id}: what to change")
