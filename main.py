"""
BlogForge Pro — Production Backend for Render
"""

import time
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
import os
import re
import html
import asyncio
from datetime import datetime
from urllib.parse import urlparse

load_dotenv()

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="BlogForge Pro", docs_url=None, redoc_url=None)

# CORS — allow your GitHub Pages domain
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://YOUR_USERNAME.github.io"  # ← CHANGE THIS
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Rate Limiter (simple in-memory, per-IP) ──────────────────────────────────


_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    window = now - 3600
    _rate_store[ip] = [t for t in _rate_store[ip] if t > window]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        return True
    _rate_store[ip].append(now)
    return False


# ── Models ────────────────────────────────────────────────────────────────────

class BlogRequest(BaseModel):
    url: str
    tone: str = "Professional"
    style: str = "Storytelling"
    days: int = Field(default=7, ge=1, le=7)
    api_key: str | None = None
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://")
        if not parsed.netloc or "." not in parsed.netloc:
            raise ValueError("Invalid URL domain")
        # Block internal/private IPs
        host = parsed.hostname or ""
        blocked = ("localhost", "127.0.0.1", "0.0.0.0",
                   "169.254", "10.", "192.168.", "172.16.")
        if any(host.startswith(b) or host == b for b in blocked):
            raise ValueError("Internal URLs are not allowed")
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in PROVIDER_MODELS:
            raise ValueError(f"Unknown provider: {v}")
        return v

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in TONE_INSTRUCTIONS:
            raise ValueError(f"Unknown tone: {v}")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        if v and len(v) > 256:
            raise ValueError("API key too long")
        return v


class BlogPlan(BaseModel):
    day: int
    title: str
    focus_keyword: str
    summary: str


class BlogPlanOutput(BaseModel):
    company_context: str = Field(
        description="Detailed summary of the website's business, audience, value proposition, and domain.")
    detected_domain: str = Field(
        description="The primary domain category.")
    plans: list[BlogPlan] = Field(
        description="Blog plans, one per requested day.")


class InlineImage(BaseModel):
    position: str
    prompt: str
    alt_text: str


class BlogPost(BaseModel):
    day: int
    title: str
    slug: str
    focus_keyword: str
    meta_description: str
    hero_image_prompt: str
    content: str
    inline_images: list[InlineImage]
    tags: list[str]
    read_time: str
    seo_score: int


class BlogOutput(BaseModel):
    blogs: list[BlogPost]


# ── Domain Strategy System ────────────────────────────────────────────────────

DOMAIN_STRATEGIES = {
    "Technology": {
        "structure": "Lead with the disruption or problem. Use data/benchmarks early. Include code snippets or architecture diagrams where relevant. End with future implications.",
        "voice": "Informed but accessible — like Wired or Ars Technica. Avoid hype. Use specific numbers.",
        "seo": "Target long-tail technical keywords. Include comparison tables.",
        "hooks": "Open with a surprising stat, a product launch impact, or a near-future scenario.",
        "references": "TechCrunch, Wired, The Verge, Ars Technica",
    },
    "Finance": {
        "structure": "Start with market context. Present thesis → evidence → counterargument → conclusion.",
        "voice": "Authoritative like Bloomberg. Be specific with numbers, dates, and sources.",
        "seo": "Target decision-stage keywords: 'should I invest in', 'X vs Y'.",
        "hooks": "Open with a market movement, a contrarian take, or a relatable money problem.",
        "references": "Bloomberg, Seeking Alpha, Investopedia, NerdWallet",
    },
    "Health": {
        "structure": "Lead with the reader's problem. Present evidence-based insights. Include actionable takeaways.",
        "voice": "Trustworthy like Healthline. Always cite research. Use empathetic but precise language.",
        "seo": "Target symptom and solution keywords. Include expert quotes.",
        "hooks": "Open with a common misconception or a new study finding.",
        "references": "Healthline, Harvard Health Blog, Well+Good",
    },
    "Travel": {
        "structure": "Open with a vivid scene. Use first-person narrative. Include practical tips woven into storytelling.",
        "voice": "Immersive like Lonely Planet. Paint pictures with words.",
        "seo": "Target 'best X in Y', 'guide to', 'things to do in' patterns.",
        "hooks": "Open with a sensory moment — a smell, a sound, a first impression.",
        "references": "Lonely Planet, Travel + Leisure, Nomadic Matt",
    },
    "Food": {
        "structure": "Lead with the dish story. Explain technique with precision. Include science where relevant.",
        "voice": "Passionate like Bon Appétit. Use sensory language.",
        "seo": "Target recipe and technique keywords.",
        "hooks": "Open with a flavor memory or a kitchen failure that led to discovery.",
        "references": "Bon Appétit, Serious Eats, NYT Cooking",
    },
    "Fashion": {
        "structure": "Lead with the cultural moment. Connect fashion to identity.",
        "voice": "Sharp and opinionated like The Cut or GQ.",
        "seo": "Target trend and 'how to style' keywords.",
        "hooks": "Open with a cultural observation or a bold fashion statement.",
        "references": "Vogue, GQ, The Cut, Harper's Bazaar",
    },
    "Marketing": {
        "structure": "Lead with the business problem. Present framework → case study → implementation steps.",
        "voice": "Strategic like HubSpot. Data-driven but readable.",
        "seo": "Target 'how to', 'strategy', 'guide' keywords.",
        "hooks": "Open with a metric that changed or a counterintuitive insight.",
        "references": "HubSpot Blog, Ahrefs Blog, Backlinko",
    },
    "Entertainment": {
        "structure": "Lead with cultural impact. Use narrative journalism style.",
        "voice": "Engaging like The Atlantic. Go beyond surface-level.",
        "seo": "Target cultural moment keywords.",
        "hooks": "Open with a scene, a quote, or a cultural observation.",
        "references": "The Atlantic, The New Yorker, Billboard",
    },
    "Education": {
        "structure": "Lead with the learning gap. Build understanding progressively.",
        "voice": "Clear like James Clear. Make complex ideas simple without dumbing down.",
        "seo": "Target 'learn', 'understand', 'explain' keywords.",
        "hooks": "Open with 'you probably think X but actually Y'.",
        "references": "James Clear, Farnam Street, Scientific American",
    },
    "General": {
        "structure": "Lead with a compelling hook. Present clear thesis. Support with evidence.",
        "voice": "Engaging and authoritative. Balance storytelling with substance.",
        "seo": "Target intent-based keywords.",
        "hooks": "Open with a surprising fact or a provocative question.",
        "references": "Medium top publications, Substack newsletters",
    },
}

TONE_INSTRUCTIONS = {
    "Professional":  "Use formal language, industry expertise, data-backed insights.",
    "Creative":      "Use vivid imagery, metaphors, storytelling, and emotional language.",
    "Casual":        "Write like a knowledgeable friend — conversational, warm, relatable.",
    "Authoritative": "Be confident, use strong statements, expert opinions.",
    "Storytelling":  "Focus on narrative arc, personal journey, emotions, vivid scenes.",
    "Inspirational": "Motivational, uplifting, powerful stories and calls to action.",
}

PROVIDER_MODELS = {
    "groq": {
        "prefix": "groq/",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
    },
    "openai": {
        "prefix": "openai/",
        "base_url": None,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "prefix": "anthropic/",
        "base_url": None,
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
    },
    "deepseek": {
        "prefix": "deepseek/",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(provider: str, model: str, api_key: str | None = None) -> LLM:
    prov = PROVIDER_MODELS[provider]  # already validated

    key = api_key
    if not key:
        env_map = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        key = os.getenv(env_map.get(provider, ""), "")

    if not key:
        raise ValueError(
            f"No API key for {provider}. Set it in environment or enter in UI.")

    full_model = f"{prov['prefix']}{model}"
    kwargs = {"model": full_model, "api_key": key}
    if prov.get("base_url"):
        kwargs["base_url"] = prov["base_url"]
    return LLM(**kwargs)


# ── Sanitization ──────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """Escape HTML entities to prevent XSS."""
    return html.escape(text, quote=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models/{provider}")
async def get_models(provider: str):
    prov = PROVIDER_MODELS.get(provider)
    if not prov:
        return JSONResponse(status_code=400, content={"error": f"Unknown provider: {provider}"})
    return JSONResponse(content={"provider": provider, "models": prov["models"]})


@app.post("/generate")
async def generate_blogs(request: BlogRequest, req: Request):
    # Rate limit by IP
    client_ip = req.client.host if req.client else "unknown"
    if is_rate_limited(client_ip):
        return JSONResponse(
            status_code=429,
            content={"success": False,
                     "error": "Rate limited. Try again later."}
        )

    try:
        llm = get_llm(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
        )
        tone_instruction = TONE_INSTRUCTIONS[request.tone]

        # ══════════════════════════════════════════
        # PHASE 1: Scrape, Detect Domain & Plan
        # ══════════════════════════════════════════
        scrape_tool = ScrapeWebsiteTool(website_url=request.url)

        planner = Agent(
            role="Senior Content Strategist & Domain Analyst",
            goal=f"Analyze {request.url}, detect its domain category, and create a strategic content calendar.",
            backstory=(
                "You are an expert at decoding websites — identifying their domain (Technology, Finance, Health, "
                "Travel, Food, Fashion, Marketing, Entertainment, Education, or General), extracting core value "
                "propositions, and planning content strategies that match the best blogs in that domain."
            ),
            tools=[scrape_tool],
            verbose=False,
            llm=llm,
        )

        plan_task = Task(
            description=(
                f"1. Scrape {request.url}.\n"
                f"2. Write a comprehensive 'company_context' summary of the business.\n"
                f"3. Detect the primary 'detected_domain' from: Technology, Finance, Health, Travel, Food, Fashion, Marketing, Entertainment, Education, General.\n"
                f"4. Create exactly {request.days} blog post plans (Day 1 to {request.days}).\n"
                f"Tone: {request.tone} | Style: {request.style}"
            ),
            expected_output=f"BlogPlanOutput with company_context, detected_domain, and {request.days} plans.",
            output_pydantic=BlogPlanOutput,
            agent=planner,
        )

        plan_crew = Crew(agents=[planner], tasks=[plan_task], verbose=False)
        plan_result = plan_crew.kickoff()
        plan_data = plan_result.pydantic
        company_context = plan_data.company_context
        domain = plan_data.detected_domain

        strategy = DOMAIN_STRATEGIES.get(domain, DOMAIN_STRATEGIES["General"])

        # Reduced delay — only wait if provider needs it (Groq free tier)
        if request.provider == "groq" and not request.api_key:
            await asyncio.sleep(12)

        # ══════════════════════════════════════════
        # PHASE 2: Write each post with domain strategy
        # ══════════════════════════════════════════
        writer = Agent(
            role="World-Class Long-Form Blogger",
            goal=f"Write comprehensive, high-quality blog posts in {request.tone} tone.",
            backstory=(
                f"You write at the level of {strategy['references']}. {tone_instruction}\n"
                f"DOMAIN STRATEGY ({domain}):\n"
                f"- Structure: {strategy['structure']}\n"
                f"- Voice: {strategy['voice']}\n"
                f"- Hooks: {strategy['hooks']}\n"
                f"- SEO: {strategy['seo']}\n"
                "Apply these strategies rigorously."
            ),
            verbose=False,
            llm=llm,
        )

        final_blogs = []

        for i, plan in enumerate(plan_data.plans):
            if i > 0 and request.provider == "groq" and not request.api_key:
                await asyncio.sleep(8)

            write_task = Task(
                description=(
                    f"Company Context: {company_context}\n"
                    f"Domain: {domain}\n\n"
                    f"Write the blog post for DAY {plan.day}.\n"
                    f"Title: {plan.title}\n"
                    f"Focus Keyword: {plan.focus_keyword}\n"
                    f"Summary: {plan.summary}\n\n"
                    f"Tone: {request.tone} | Style: {request.style}\n\n"
                    f"DOMAIN STRATEGY ({domain} — like {strategy['references']}):\n"
                    f"- Hook: {strategy['hooks']}\n"
                    f"- Structure: {strategy['structure']}\n"
                    f"- Voice: {strategy['voice']}\n"
                    f"- SEO: {strategy['seo']}\n\n"
                    "STRICT REQUIREMENTS:\n"
                    "- content: MINIMUM 1500 words. At least 6 sections with ## H2 headings.\n"
                    "  Each section: 3-5 detailed paragraphs. Do NOT summarize — write in full.\n"
                    "  Must include Introduction, 4+ body sections, Conclusion.\n"
                    "  Use ### H3 subheadings. Expand every point with examples, data, narrative.\n"
                    "- inline_images: exactly 4 objects with position, prompt (3-5 concrete nouns), alt_text.\n"
                    "- hero_image_prompt: short noun phrase.\n"
                    "- slug: lowercase hyphen-separated.\n"
                    "- seo_score: integer 65-95.\n"
                    "- read_time: based on actual word count.\n"
                    "Output ONLY the JSON. Do not truncate content."
                ),
                expected_output="Complete BlogPost JSON with 1500+ word content.",
                output_pydantic=BlogPost,
                agent=writer,
            )

            write_crew = Crew(agents=[writer], tasks=[
                              write_task], verbose=False)
            write_result = write_crew.kickoff()

            blog = write_result.pydantic
            blog.day = plan.day
            final_blogs.append(blog)

        return JSONResponse(content={
            "success": True,
            "domain": domain,
            "strategy_ref": strategy["references"],
            "blogs": [b.model_dump() for b in final_blogs],
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
