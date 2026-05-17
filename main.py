"""
BlogForge Pro — Production Backend for Render
v2: Dual input mode (URL scrape + Topic/thought research)
Fix: Raw JSON output parsing to bypass Groq function-calling length limits
"""

import time
import json
import re
import html
import asyncio
import os
import traceback
from collections import defaultdict
import logging
from datetime import datetime
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BlogForge")

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="BlogForge Pro", docs_url=None, redoc_url=None)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,https://blogforge-3dvj.onrender.com,https://arthist03.github.io"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Rate Limiter ──────────────────────────────────────────────────────────────

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
    url: str | None = None
    topic: str | None = None
    tone: str = "Professional"
    style: str = "Storytelling"
    days: int = Field(default=7, ge=1, le=7)
    api_key: str | None = None
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    base_url: str | None = None
    style_profile: dict | None = None
    preset: str | None = None
    preset_text: str | None = None
    ensemble: bool = True
    rescore: bool = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://")
        if not parsed.netloc or "." not in parsed.netloc:
            raise ValueError("Invalid URL domain")
        host = parsed.hostname or ""
        blocked = ("localhost", "127.0.0.1", "0.0.0.0",
                   "169.254", "10.", "192.168.", "172.16.")
        if any(host.startswith(b) or host == b for b in blocked):
            raise ValueError("Internal URLs are not allowed")
        return v

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        if len(v) > 5000:
            raise ValueError("Topic text is too long (max 5000 characters)")
        if len(v) < 3:
            raise ValueError("Topic is too short — give at least a few words")
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in PROVIDER_MODELS and v != "custom":
            raise ValueError(f"Unknown provider: {v}")
        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Custom base URL must be a valid http(s) URL")
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

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v not in PRESET_PROMPTS and v != "custom":
            raise ValueError(f"Unknown preset: {v}")
        return v


class StyleExtractRequest(BaseModel):
    text: str | None = None
    urls: list[str] | None = None
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: str | None = None
    base_url: str | None = None


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


class TopicPlanOutput(BaseModel):
    topic_context: str = Field(
        description="Deep research summary of the topic — key facts, trends, audience, angles, and subtopics.")
    detected_domain: str = Field(
        description="The primary domain category.")
    plans: list[BlogPlan] = Field(
        description="Blog plans, one per requested day.")


class InlineImage(BaseModel):
    position: str = ""
    prompt: str = ""
    alt_text: str = ""


class BlogPost(BaseModel):
    day: int = 1
    title: str = ""
    slug: str = ""
    focus_keyword: str = ""
    meta_description: str = ""
    hero_image_prompt: str = ""
    content: str = ""
    inline_images: list[InlineImage] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    read_time: str = ""
    seo_score: int = 95


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

# ── Prompt Presets ──────────────────────────────────────────────────────────
PRESET_PROMPTS = {
    "balanced": (
        "You are a balanced, high-quality content engine. Prioritize accuracy, depth, and readability. "
        "Produce well-researched, evidence-backed sections, and ensure the output is optimized for search intent."
    ),
    "seo": (
        "You are an SEO expert. Prioritize keywords, headings, meta descriptions, and on-page structure. "
        "Aim for high topical authority and include internal linking suggestions where appropriate."
    ),
    "creative": (
        "You are a creative storyteller. Emphasize vivid imagery, narrative flow, and emotional resonance. "
        "Use metaphors and examples to engage readers while preserving clarity."
    ),
    "researcher": (
        "You are an expert researcher and analyst. Provide data-backed arguments, cite likely sources, "
        "and synthesize multiple perspectives into a clear narrative. Prioritize depth over breath."
    ),
    "concise": (
        "You are a concise editor. Keep prose tight, prioritize clarity, and avoid unnecessary repetition. "
        "Produce punchy, actionable sections ideal for busy readers."
    ),
    "journalist": (
        "You are an investigative journalist. Lead with the most important facts, include quotes and context, "
        "and maintain strong sourcing and neutral tone unless the brief asks for opinion."
    ),
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
        "models": [
            "gpt-5.5",
            "gpt-5.2",
            "o3",
            "o4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    },
    "anthropic": {
        "prefix": "anthropic/",
        "base_url": None,
        "models": [
            "claude-opus-4.7",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ],
    },
    "google": {
        "prefix": "google/",
        "base_url": None,
        "models": ["gemini-3-pro", "gemini-3.1-pro", "gemini-flash"],
    },
    "xai": {
        "prefix": "grok/",
        "base_url": None,
        "models": ["grok-4", "grok-4.1"],
    },
    "qwen": {
        "prefix": "qwen/",
        "base_url": None,
        "models": ["qwen3", "qwen3.5"],
    },
    "mistral": {
        "prefix": "mistral/",
        "base_url": None,
        "models": ["mistral-large-2"],
    },
    "deepseek": {
        "prefix": "deepseek/",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}

# ── Ensemble candidates (ordered fallbacks to try alongside the primary model)
ENSEMBLE_CANDIDATES = {
    "default": [
        ("openai", "o4"),
        ("anthropic", "claude-opus-4.7"),
        ("groq", "llama-3.3-70b-versatile"),
    ],
    "openai": [("openai", "o4"), ("anthropic", "claude-opus-4.7"), ("groq", "llama-3.3-70b-versatile")],
    "anthropic": [("anthropic", "claude-opus-4.7"), ("openai", "o4"), ("groq", "llama-3.3-70b-versatile")],
    "groq": [("groq", "llama-3.3-70b-versatile"), ("openai", "o4"), ("anthropic", "claude-opus-4.7")],
    "deepseek": [("deepseek", "deepseek-chat"), ("openai", "o4"), ("groq", "llama-3.3-70b-versatile")],
}

SCRAPE_ARTIFACT_PHRASES = [
    "The following text is scraped website content",
    "please enable JavaScript and disable ad blockers",
]


def sanitize_source_context(text: str) -> str:
    """Remove common scrape boilerplate so the writer doesn't copy it into output."""
    if not text:
        return ""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue
        lower = s.lower()
        if any(p.lower() in lower for p in SCRAPE_ARTIFACT_PHRASES):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_blog_post(blog: BlogPost, domain: str, focus_keyword: str) -> BlogPost:
    """Clean leaked scrape artifacts and strengthen weak image prompts."""
    if blog.content:
        content = blog.content
        for p in SCRAPE_ARTIFACT_PHRASES:
            content = re.sub(re.escape(p), "", content, flags=re.IGNORECASE)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        blog.content = content

    # Upgrade weak hero prompts so image generation has enough scene detail
    hp = (blog.hero_image_prompt or "").strip()
    if len(hp.split()) < 6:
        base = focus_keyword or blog.title or domain or "technology article"
        blog.hero_image_prompt = (
            f"{base}, editorial scene, cinematic lighting, high detail, professional composition"
        )

    if blog.inline_images:
        for img in blog.inline_images:
            prompt = (img.prompt or "").strip()
            if len(prompt.split()) < 5:
                anchor = focus_keyword or domain or "business"
                img.prompt = f"{anchor}, realistic setting, people collaborating, modern workspace, high detail"
            if not (img.alt_text or "").strip():
                img.alt_text = f"Illustration related to {focus_keyword or blog.title or 'the article topic'}"

    return blog


def score_blog_local(blog: BlogPost) -> float:
    """Heuristic scorer for generated blog posts. Higher is better.

    Factors: word count, H2 headings count, seo_score field, presence of hero image prompt and inline images.
    """
    try:
        content = blog.content or ""
        word_count = max(len(content.split()), 0)
        wc_score = min(word_count / 2000.0, 1.0)
        h2_count = content.count("## ")
        h2_score = min(h2_count / 6.0, 1.0)
        seo_score = getattr(blog, "seo_score", 50) or 50
        seo_score_norm = min(max((seo_score - 50) / 50.0, 0.0), 1.0)
        hero = 1.0 if (getattr(blog, "hero_image_prompt", "") or "") else 0.0
        inline_images = len(getattr(blog, "inline_images", []) or [])
        inline_score = min(inline_images / 4.0, 1.0)

        artifact_penalty = 0.0
        lower_content = content.lower()
        if any(p.lower() in lower_content for p in SCRAPE_ARTIFACT_PHRASES):
            artifact_penalty = 0.25

        completeness = 0
        fields = [blog.title, blog.slug, blog.focus_keyword,
                  blog.meta_description, blog.content]
        completeness = sum(1 for f in fields if f) / len(fields)

        score = (
            0.40 * wc_score
            + 0.20 * h2_score
            + 0.20 * seo_score_norm
            + 0.10 * inline_score
            + 0.05 * hero
            + 0.05 * completeness
            - artifact_penalty
        )
        return float(score)
    except Exception:
        return 0.0


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    prov = PROVIDER_MODELS.get(provider)
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

    if provider == "custom":
        full_model = model
        kwargs = {"model": full_model, "api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        return LLM(**kwargs)

    full_model = f"{prov['prefix']}{model}"
    kwargs = {"model": full_model, "api_key": key}
    if prov.get("base_url"):
        kwargs["base_url"] = prov["base_url"]
    return LLM(**kwargs)


# ── JSON Extraction Helpers ───────────────────────────────────────────────────

def extract_json_from_text(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def compute_read_time(content: str) -> str:
    if not content:
        return "1 min read"
    word_count = len(content.split())
    minutes = max(1, round(word_count / 250))
    return f"{minutes} min read"


def parse_blog_from_raw(raw_text: str, day: int) -> BlogPost:
    data = extract_json_from_text(raw_text)
    if data and isinstance(data, dict):
        images_raw = data.get("inline_images", [])
        images = []
        for img in images_raw:
            if isinstance(img, dict):
                images.append(InlineImage(
                    position=str(img.get("position", "")),
                    prompt=str(img.get("prompt", "")),
                    alt_text=str(img.get("alt_text", "")),
                ))
        content = data.get("content", "")
        tags_raw = data.get("tags", [])
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
        read_time = compute_read_time(content)
        try:
            seo_score = int(data.get("seo_score", 95))
            seo_score = max(90, min(100, seo_score))
        except (ValueError, TypeError):
            seo_score = 95
        return BlogPost(
            day=day,
            title=str(data.get("title", f"Blog Post Day {day}")),
            slug=str(data.get("slug", f"blog-post-day-{day}")),
            focus_keyword=str(data.get("focus_keyword", "")),
            meta_description=str(data.get("meta_description", "")),
            hero_image_prompt=str(data.get("hero_image_prompt", "")),
            content=content,
            inline_images=images,
            tags=tags,
            read_time=read_time,
            seo_score=seo_score,
        )

    logger.warning(f"Day {day}: JSON parse failed, using raw text as content")
    content = raw_text.strip() if raw_text else ""
    return BlogPost(
        day=day,
        title=f"Blog Post — Day {day}",
        slug=f"blog-post-day-{day}",
        content=content,
        inline_images=[],
        tags=[],
        read_time=compute_read_time(content),
        seo_score=95,
    )


# ── Blog JSON Schema ─────────────────────────────────────────────────────────

BLOG_JSON_SCHEMA = """{
  "day": <integer>,
  "title": "<string>",
  "slug": "<lowercase-hyphenated-string>",
  "focus_keyword": "<string>",
  "meta_description": "<string, 150-160 chars>",
  "hero_image_prompt": "<short noun phrase>",
  "content": "<string — full markdown blog content. CRITICAL: You MUST escape all newlines as \\n\\n within this string. DO NOT use raw unescaped newlines>",
  "inline_images": [
    {"position": "<top|middle|bottom|side>", "prompt": "<3-5 concrete nouns>", "alt_text": "<descriptive>"},
    {"position": "...", "prompt": "...", "alt_text": "..."}
  ],
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "read_time": "<N min read>",
  "seo_score": <integer 93-99>
}"""


# ── Planner: URL Mode ─────────────────────────────────────────────────────────

def run_url_planner(llm: LLM, url: str, days: int, tone: str, style: str) -> tuple[str, str, list[BlogPlan]]:
    scrape_tool = ScrapeWebsiteTool(website_url=url)

    planner = Agent(
        role="Senior Content Strategist & Domain Analyst",
        goal=f"Analyze {url}, detect its domain category, and create a strategic content calendar.",
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
            f"1. Scrape {url}.\n"
            f"2. Write a comprehensive 'company_context' summary of the business.\n"
            f"3. Detect the primary 'detected_domain' from: Technology, Finance, Health, Travel, Food, Fashion, Marketing, Entertainment, Education, General.\n"
            f"4. Create exactly {days} blog post plans (Day 1 to {days}).\n"
            f"Tone: {tone} | Style: {style}"
        ),
        expected_output=f"BlogPlanOutput with company_context, detected_domain, and {days} plans.",
        output_pydantic=BlogPlanOutput,
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[plan_task], verbose=False)
    result = crew.kickoff()
    data = result.pydantic
    return data.company_context, data.detected_domain, data.plans


# ── Planner: Topic Mode ───────────────────────────────────────────────────────

def run_topic_planner(llm: LLM, topic: str, days: int, tone: str, style: str) -> tuple[str, str, list[BlogPlan]]:
    planner = Agent(
        role="Senior Content Strategist & Topic Research Analyst",
        goal="Deep-research the given topic, detect its domain category, and create a strategic content calendar.",
        backstory=(
            "You are a world-class content strategist and researcher. Given a topic or a user's detailed thoughts, "
            "you conduct deep research using your knowledge — identifying the domain (Technology, Finance, Health, "
            "Travel, Food, Fashion, Marketing, Entertainment, Education, or General), understanding the target "
            "audience, key trends, expert perspectives, common questions, and content gaps. "
            "You plan content series that are comprehensive, SEO-optimized, and match the best publications in the domain."
        ),
        verbose=False,
        llm=llm,
    )

    plan_task = Task(
        description=(
            f"USER INPUT (topic/thoughts):\n"
            f"---\n{topic}\n---\n\n"
            f"Based on the above, do the following:\n"
            f"1. Write a comprehensive 'topic_context' research summary (300-500 words). Include:\n"
            f"   - What the topic is about and why it matters now\n"
            f"   - Target audience and their pain points\n"
            f"   - Key trends, data points, and expert perspectives\n"
            f"   - Common questions people ask about this topic\n"
            f"   - Content gaps that exist in current online coverage\n"
            f"   - If the user shared specific thoughts/opinions, incorporate and expand on them\n"
            f"2. Detect the primary 'detected_domain' from: Technology, Finance, Health, Travel, Food, Fashion, Marketing, Entertainment, Education, General.\n"
            f"3. Create exactly {days} blog post plans (Day 1 to {days}). Each plan should:\n"
            f"   - Cover a distinct angle or subtopic\n"
            f"   - Build on the user's thoughts if they provided detailed input\n"
            f"   - Have a specific, SEO-friendly focus_keyword\n"
            f"   - Avoid redundancy across days\n"
            f"Tone: {tone} | Style: {style}"
        ),
        expected_output=f"TopicPlanOutput with topic_context, detected_domain, and {days} plans.",
        output_pydantic=TopicPlanOutput,
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[plan_task], verbose=False)
    result = crew.kickoff()
    data = result.pydantic
    return data.topic_context, data.detected_domain, data.plans


# ── Shared Writer Pipeline ────────────────────────────────────────────────────

async def write_blogs(
    llm: LLM,
    context: str,
    domain: str,
    strategy: dict,
    plans: list[BlogPlan],
    tone: str,
    style: str,
    provider: str,
    has_api_key: bool,
    input_mode: str,
    style_profile: dict | None,
    preset_instruction: str | None = None,
    ensemble_enabled: bool = True,
    rescore_enabled: bool = True,
) -> list[BlogPost]:
    context_label = "Company Context" if input_mode == "url" else "Topic Research Context"
    safe_context = sanitize_source_context(context)

    # Merge preset instruction into the writer backstory when provided
    backstory_prefix = preset_instruction + "\n\n" if preset_instruction else ""

    style_instruction_global = ""
    if style_profile:
        style_instruction_global = (
            "\n*** CRITICAL USER STYLE OVERRIDES ***\n"
            "You MUST rigidly follow this extracted writing style profile, overriding any default tone guidelines.\n"
            f"- Tone: {style_profile.get('tone', '')}\n"
            f"- Perspective: {style_profile.get('person', '')} person\n"
            f"- Sentence length: {style_profile.get('avg_sentence_length', '')}\n"
            f"- Vocabulary: {style_profile.get('vocabulary_register') or style_profile.get('vocabulary', '')}\n"
            f"- Use headers: {style_profile.get('uses_headers', '')}\n"
            f"- Use bullet points: {style_profile.get('uses_bullet_points', '')}\n"
            f"- Opening pattern: {style_profile.get('opening_pattern', '')}\n"
            f"- Closing pattern: {style_profile.get('closing_pattern', '')}\n"
            f"- Specific habits to replicate: {', '.join(style_profile.get('unique_patterns', []))}\n"
            "Do NOT reference that you are using this style profile. Just adopt it completely.\n"
            "*** END STYLE PROFILE ***\n"
        )

    writer = Agent(
        role="World-Class Long-Form Blogger",
        goal=f"Write comprehensive, high-quality blog posts. Always output valid JSON only.",
        backstory=(
            f"{backstory_prefix}You write at the level of {strategy['references']}. "
            f"{TONE_INSTRUCTIONS[tone] if not style_profile else ''}\n"
            f"DOMAIN STRATEGY ({domain}):\n"
            f"- Structure: {strategy['structure']}\n"
            f"- Voice: {strategy['voice']}\n"
            f"- Hooks: {strategy['hooks']}\n"
            f"- SEO: {strategy['seo']}\n"
            f"Apply these strategies rigorously.\n"
            f"{style_instruction_global}"
            "CRITICAL: You ALWAYS respond with ONLY a valid JSON object. No markdown fences, no commentary — just JSON."
        ),
        verbose=False,
        llm=llm,
    )

    final_blogs = []
    MAX_RETRIES = 2

    for i, plan in enumerate(plans):
        if i > 0 and provider == "groq" and not has_api_key:
            await asyncio.sleep(8)

        blog = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                write_task = Task(
                    description=(
                        f"{context_label}: {safe_context}\n"
                        f"Domain: {domain}\n\n"
                        f"Write the blog post for DAY {plan.day}.\n"
                        f"Title: {plan.title}\n"
                        f"Focus Keyword: {plan.focus_keyword}\n"
                        f"Summary: {plan.summary}\n\n"
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
                        "- inline_images: exactly 4 objects with position, prompt, alt_text.\n"
                        "  Each image prompt must describe a concrete visual scene in 8-16 words (subject + setting + visual style).\n"
                        "  Avoid generic prompts like 'AI system' or 'technology'.\n"
                        "- hero_image_prompt: 10-18 words with a specific visual scene, not a generic phrase.\n"
                        "- slug: lowercase hyphen-separated.\n"
                        "- seo_score: integer 93-99.\n"
                        "- read_time: estimate based on word count.\n\n"
                        "CONTENT SAFETY / CLEANLINESS RULES:\n"
                        "- DO NOT include source artifacts such as 'The following text is scraped website content'.\n"
                        "- DO NOT include website boilerplate like 'please enable JavaScript and disable ad blockers'.\n"
                        "- Write an original article synthesis. Do not paste raw scraped snippets.\n\n"
                        "CRITICAL JSON FORMATTING RULES:\n"
                        "1. Respond with ONLY a valid JSON object matching the exact schema.\n"
                        "2. No markdown code fences (do not wrap in ```json).\n"
                        "3. No extra text or opening/closing explanations.\n"
                        "4. Inside the 'content' field, you MUST escape every single line break and paragraph break using literal \\n or \\n\\n. NEVER use raw/real newlines inside the JSON string, or it will crash the system.\n\n"
                        f"{BLOG_JSON_SCHEMA}"
                    ),
                    expected_output="A single valid JSON object with perfectly escaped strings. You must map your response precisely so that it perfectly deserializes into the provided Blog schema structure.",
                    output_pydantic=BlogPost,
                    agent=writer,
                )

                # Ensemble: gather candidate outputs from multiple LLMs
                candidates: list[tuple[BlogPost, float, str]] = []

                # primary candidate
                try:
                    write_crew = Crew(agents=[writer], tasks=[
                                      write_task], verbose=False)
                    write_result = write_crew.kickoff()
                    if hasattr(write_result, 'pydantic') and write_result.pydantic:
                        candidate = write_result.pydantic
                    else:
                        raw_text = write_result.raw if hasattr(
                            write_result, 'raw') else str(write_result)
                        candidate = parse_blog_from_raw(raw_text, plan.day)
                    if candidate:
                        candidate.day = plan.day
                        sc = score_blog_local(candidate)
                        candidates.append(
                            (candidate, sc, f"primary:{provider}/{candidate.title[:40]}"))
                except Exception as e:
                    logger.debug(f"Primary candidate failed: {e}")

                # additional ensemble candidates
                if ensemble_enabled:
                    ens = ENSEMBLE_CANDIDATES.get(
                        provider, ENSEMBLE_CANDIDATES.get('default', []))
                    # limit to 3 extra candidates
                    for (pprov, pmodel) in ens[:3]:
                        try:
                            if pprov == provider and pmodel == writer.llm.model:
                                continue
                            alt_llm = get_llm(pprov, pmodel, api_key=None)
                            alt_writer = Agent(
                                role=writer.role,
                                goal=writer.goal,
                                backstory=writer.backstory,
                                verbose=False,
                                llm=alt_llm,
                            )
                            alt_task = Task(
                                description=write_task.description,
                                expected_output=write_task.expected_output,
                                output_pydantic=write_task.output_pydantic,
                                agent=alt_writer,
                            )
                            alt_crew = Crew(agents=[alt_writer], tasks=[
                                            alt_task], verbose=False)
                            alt_result = alt_crew.kickoff()
                            if hasattr(alt_result, 'pydantic') and alt_result.pydantic:
                                alt_blog = alt_result.pydantic
                            else:
                                raw_text = alt_result.raw if hasattr(
                                    alt_result, 'raw') else str(alt_result)
                                alt_blog = parse_blog_from_raw(
                                    raw_text, plan.day)
                            if alt_blog:
                                alt_blog.day = plan.day
                                sc = score_blog_local(alt_blog)
                                candidates.append(
                                    (alt_blog, sc, f"ens:{pprov}/{pmodel}"))
                        except Exception as e:
                            logger.debug(
                                f"Ensemble candidate {pprov}/{pmodel} failed: {e}")

                # choose best candidate by score
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    blog = candidates[0][0]
                    blog = sanitize_blog_post(blog, domain, plan.focus_keyword)
                    best_score = candidates[0][1]
                    logger.info(
                        f"[ENSEMBLE] Day {plan.day} best score={best_score:.3f} source={candidates[0][2]}")
                else:
                    blog = None

                if blog and blog.content and len(blog.content.split()) > 900:
                    break
                else:
                    logger.warning(
                        f"[WARN] Day {plan.day} attempt {attempt+1}: content too short or empty (minimum 900 words required)")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(5)

            except Exception as e:
                logger.error(
                    f"[ERROR] Day {plan.day} attempt {attempt+1}: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(5)
                else:
                    blog = BlogPost(
                        day=plan.day,
                        title=plan.title,
                        slug=plan.title.lower().replace(" ", "-")[:60],
                        focus_keyword=plan.focus_keyword,
                        meta_description=plan.summary[:160],
                        hero_image_prompt=plan.focus_keyword,
                        content=f"## {plan.title}\n\n{plan.summary}\n\n*Content generation encountered an error. Please try regenerating this post.*",
                        inline_images=[],
                        tags=[plan.focus_keyword],
                        read_time="1 min read",
                        seo_score=50,
                    )

        if blog is None:
            blog = BlogPost(
                day=plan.day,
                title=plan.title,
                slug=f"day-{plan.day}",
                focus_keyword=plan.focus_keyword,
                meta_description=plan.summary[:160],
                content=f"## {plan.title}\n\n{plan.summary}",
                tags=[],
                read_time="1 min read",
                seo_score=50,
            )

        blog.read_time = compute_read_time(blog.content)
        final_blogs.append(blog)

    return final_blogs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models/{provider}")
async def get_models(provider: str):
    prov = PROVIDER_MODELS.get(provider)
    if not prov:
        return JSONResponse(status_code=400, content={"error": f"Unknown provider: {provider}"})
    return JSONResponse(content={"provider": provider, "models": prov["models"]})


@app.post("/extract-style")
async def extract_style(request: StyleExtractRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    if is_rate_limited(client_ip):
        return JSONResponse(
            status_code=429,
            content={"success": False,
                     "error": "Rate limited. Try again later."}
        )

    try:
        combined_text = (request.text or "")

        if request.urls:
            for url in request.urls:
                if not url.strip():
                    continue
                try:
                    tool = ScrapeWebsiteTool(website_url=url.strip())
                    scraped = tool.run()
                    combined_text += f"\n\n--- Content from {url} ---\n{str(scraped)[:4000]}"
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")

        if len(combined_text.strip()) < 50:
            return JSONResponse(status_code=400, content={"success": False, "error": "Not enough content provided to analyze style."})

        llm = get_llm(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
        )

        extractor = Agent(
            role="Writing Style Analyst",
            goal="Analyze the writing style of the provided text and return ONLY valid JSON.",
            backstory="You are a linguistic expert who analyzes text. You always respond in raw JSON.",
            verbose=False,
            llm=llm,
        )

        extract_task = Task(
            description=(
                "Analyze the writing style of the following blog posts and return a JSON object with these fields:\n"
                "tone (e.g. conversational, technical, opinionated), person (first/second/third), avg_sentence_length (short/medium/long), "
                "uses_headers (true/false), uses_bullet_points (true/false), opening_pattern (one sentence describing how this writer opens posts), "
                "closing_pattern (one sentence describing how this writer ends posts), vocabulary_register (casual/semi-formal/formal), "
                "unique_patterns (list of 2-3 specific stylistic habits). Return only valid JSON. No explanation.\n\n"
                f"TEXT:\n{combined_text}"
            ),
            expected_output="JSON object containing style profile details.",
            agent=extractor,
        )

        crew = Crew(agents=[extractor], tasks=[extract_task], verbose=False)
        result = crew.kickoff()
        data = extract_json_from_text(result.raw)

        if not data:
            return JSONResponse(status_code=400, content={"success": False, "error": "Failed to extract style JSON from model output."})

        return JSONResponse(content={"success": True, "style_profile": data})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/generate")
async def generate_blogs(request: BlogRequest, req: Request):
    if not request.url and not request.topic:
        return JSONResponse(
            status_code=400,
            content={"success": False,
                     "error": "Provide either a URL or a topic to generate blogs."}
        )

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
            base_url=request.base_url,
        )

        # ══════════════════════════════════════════
        # PHASE 1: Plan — branch on input mode
        # ══════════════════════════════════════════
        if request.url:
            input_mode = "url"
            context, domain, plans = run_url_planner(
                llm, request.url, request.days, request.tone, request.style
            )
        else:
            input_mode = "topic"
            context, domain, plans = run_topic_planner(
                llm, request.topic, request.days, request.tone, request.style
            )

        strategy = DOMAIN_STRATEGIES.get(domain, DOMAIN_STRATEGIES["General"])

        if request.provider == "groq" and not request.api_key:
            await asyncio.sleep(12)

        # ══════════════════════════════════════════
        # PHASE 2: Write — shared pipeline
        # ══════════════════════════════════════════
        # Build preset instruction if provided
        preset_instruction = None
        if request.preset and request.preset != "custom":
            preset_instruction = PRESET_PROMPTS.get(request.preset)
        elif request.preset == "custom" and request.preset_text:
            preset_instruction = request.preset_text

        final_blogs = await write_blogs(
            llm=llm,
            context=context,
            domain=domain,
            strategy=strategy,
            plans=plans,
            tone=request.tone,
            style=request.style,
            provider=request.provider,
            has_api_key=bool(request.api_key),
            input_mode=input_mode,
            style_profile=request.style_profile,
            preset_instruction=preset_instruction,
        )

        return JSONResponse(content={
            "success": True,
            "domain": domain,
            "input_mode": input_mode,
            "strategy_ref": strategy["references"],
            "blogs": [b.model_dump() for b in final_blogs],
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
