from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
import os
import re
import asyncio
from datetime import datetime

load_dotenv()
app = FastAPI(title="BlogForge Pro")

# ── Models ────────────────────────────────────────────────────────────────────


class BlogRequest(BaseModel):
    url: str
    tone: str = "Professional"
    style: str = "Storytelling"
    days: int = Field(default=7, ge=1, le=7)
    api_key: str | None = None       # User's own API key
    provider: str = "groq"           # groq | openai | anthropic | deepseek
    model: str = "llama-3.3-70b-versatile"


class BlogPlan(BaseModel):
    day: int
    title: str
    focus_keyword: str
    summary: str


class BlogPlanOutput(BaseModel):
    company_context: str = Field(
        description="Detailed summary of the website's business, audience, value proposition, and domain.")
    detected_domain: str = Field(
        description="The primary domain category (e.g., Technology, Finance, Health, Travel, Food, Fashion, Marketing, Entertainment, Education, General).")
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
# Derived from analyzing world-class blogs: TechCrunch, HBR, Wired, Healthline,
# Bon Appétit, Lonely Planet, HubSpot, Farnam Street, The Atlantic, etc.

DOMAIN_STRATEGIES = {
    "Technology": {
        "structure": "Lead with the disruption or problem. Use data/benchmarks early. Include code snippets or architecture diagrams where relevant. End with future implications.",
        "voice": "Informed but accessible — like Wired or Ars Technica. Avoid hype. Use specific numbers, version names, and comparisons. Show don't tell.",
        "seo": "Target long-tail technical keywords. Include comparison tables. Use 'How to', 'vs', 'Best' patterns.",
        "hooks": "Open with a surprising stat, a product launch impact, or a 'what if' scenario about near-future tech.",
        "references": "TechCrunch, Wired, The Verge, Ars Technica",
    },
    "Finance": {
        "structure": "Start with the market context or economic signal. Present thesis → evidence → counterargument → conclusion. Use charts/data references heavily.",
        "voice": "Authoritative like Bloomberg or Seeking Alpha. Be specific with numbers, dates, and sources. Avoid vague optimism.",
        "seo": "Target decision-stage keywords: 'should I invest in', 'X vs Y', 'best X for Y'. Include calculators or comparison frameworks.",
        "hooks": "Open with a market movement, a contrarian take, or a relatable money problem.",
        "references": "Bloomberg, Seeking Alpha, Investopedia, NerdWallet",
    },
    "Health": {
        "structure": "Lead with the reader's problem/symptom. Present evidence-based insights with source citations. Include actionable takeaways in each section.",
        "voice": "Trustworthy like Healthline or Harvard Health. Always cite research. Use empathetic but precise language. Avoid medical claims without qualification.",
        "seo": "Target symptom and solution keywords. Use 'According to research' patterns. Include expert quotes.",
        "hooks": "Open with a common misconception, a new study finding, or a relatable health scenario.",
        "references": "Healthline, Harvard Health Blog, Well+Good, mindbodygreen",
    },
    "Travel": {
        "structure": "Open with a vivid scene (sensory details). Use first-person narrative. Include practical tips (cost, timing, logistics) woven into storytelling.",
        "voice": "Immersive like Lonely Planet or Travel + Leisure. Paint pictures with words. Balance wanderlust with practical utility.",
        "seo": "Target 'best X in Y', 'guide to', 'things to do in' patterns. Include budget breakdowns and seasonal tips.",
        "hooks": "Open with a sensory moment — a smell, a sound, a first impression of a place.",
        "references": "Lonely Planet, Travel + Leisure, Nomadic Matt, The Points Guy",
    },
    "Food": {
        "structure": "Lead with the dish/ingredient story. Explain technique with precision. Include science where relevant (like Serious Eats). End with variations and tips.",
        "voice": "Passionate like Bon Appétit. Use sensory language. Be specific about techniques, temperatures, timing. Tell the story behind the food.",
        "seo": "Target recipe and technique keywords. Use structured data concepts. Include ingredient lists and timing.",
        "hooks": "Open with a flavor memory, a kitchen failure that led to discovery, or a cultural food story.",
        "references": "Bon Appétit, Serious Eats, NYT Cooking",
    },
    "Fashion": {
        "structure": "Lead with the cultural moment or trend signal. Connect fashion to identity and culture. Include specific product/brand references with context.",
        "voice": "Sharp and opinionated like The Cut or GQ. Take a stance. Blend high and low. Be culturally aware.",
        "seo": "Target trend and 'how to style' keywords. Seasonal content. Include brand names and price points.",
        "hooks": "Open with a cultural observation, a street style moment, or a bold fashion statement.",
        "references": "Vogue, GQ, The Cut, Harper's Bazaar",
    },
    "Marketing": {
        "structure": "Lead with the business problem. Present framework → case study → implementation steps. Include specific metrics and ROI data.",
        "voice": "Strategic like HubSpot or Backlinko. Data-driven but readable. Include actionable frameworks. Show real results.",
        "seo": "Target 'how to', 'strategy', 'guide' keywords. Include statistics and case studies. Use numbered frameworks.",
        "hooks": "Open with a metric that changed, a failed campaign lesson, or a counterintuitive marketing insight.",
        "references": "HubSpot Blog, Ahrefs Blog, Backlinko, Search Engine Land",
    },
    "Entertainment": {
        "structure": "Lead with the cultural impact or emotional connection. Use narrative journalism style. Include context, history, and cultural analysis.",
        "voice": "Engaging like The Atlantic or The New Yorker. Go beyond surface-level. Find the story within the story.",
        "seo": "Target cultural moment keywords. Include trending names and titles. Use 'review', 'analysis', 'impact' patterns.",
        "hooks": "Open with a scene, a quote, or a cultural observation that captures the zeitgeist.",
        "references": "The Atlantic, The New Yorker, Billboard, Airbnb News",
    },
    "Education": {
        "structure": "Lead with the learning gap or misconception. Build understanding progressively. Use examples at every level. End with practice/application.",
        "voice": "Clear like James Clear or Farnam Street. Make complex ideas simple without dumbing down. Use analogies and mental models.",
        "seo": "Target 'learn', 'understand', 'explain', 'guide' keywords. Include step-by-step progressions.",
        "hooks": "Open with a 'you probably think X but actually Y' pattern or a relatable learning struggle.",
        "references": "James Clear, Farnam Street, Scientific American, Nautilus",
    },
    "General": {
        "structure": "Lead with a compelling hook. Present clear thesis. Support with evidence and examples. End with actionable insight or thought-provoking conclusion.",
        "voice": "Engaging and authoritative. Balance storytelling with substance. Be specific, avoid generic statements.",
        "seo": "Target intent-based keywords. Use clear H2/H3 structure. Include data and examples.",
        "hooks": "Open with a surprising fact, a personal anecdote, or a provocative question.",
        "references": "Medium top publications, Substack newsletters",
    },
}


# ── Tone Instructions ─────────────────────────────────────────────────────────

TONE_INSTRUCTIONS = {
    "Professional":  "Use formal language, industry expertise, data-backed insights, and authoritative tone.",
    "Creative":      "Use vivid imagery, metaphors, storytelling, and emotional language.",
    "Casual":        "Write like a knowledgeable friend — conversational, warm, and relatable.",
    "Authoritative": "Be confident, use strong statements, expert opinions, and leadership tone.",
    "Storytelling":  "Focus on narrative arc, personal journey, emotions, and vivid scenes.",
    "Inspirational": "Motivational, uplifting, use powerful stories and calls to action.",
}

TONE_COLORS = {
    "Creative":      ("hsl(350,85%,55%)",  "hsl(350,85%,48%)"),
    "Professional":  ("hsl(220,15%,10%)",  "hsl(220,15%,20%)"),
    "Authoritative": ("hsl(230,60%,28%)",  "hsl(230,60%,38%)"),
    "Casual":        ("hsl(25,95%,52%)",   "hsl(25,95%,42%)"),
    "Storytelling":  ("hsl(262,70%,50%)",  "hsl(262,70%,40%)"),
    "Inspirational": ("hsl(158,72%,36%)",  "hsl(158,72%,28%)"),
}

# ── Provider → Model mapping for frontend ────────────────────────────────────

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
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ],
    },
    "deepseek": {
        "prefix": "deepseek/",
        "base_url": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
}


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(provider: str = "groq", model: str = "llama-3.3-70b-versatile", api_key: str | None = None) -> LLM:
    prov = PROVIDER_MODELS.get(provider)
    if not prov:
        raise ValueError(f"Unknown provider: {provider}")

    # Resolve API key: user-provided > env variable
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
            f"No API key provided for {provider}. Set it in .env or enter it in the UI.")

    full_model = f"{prov['prefix']}{model}"
    kwargs = {"model": full_model, "api_key": key}
    if prov.get("base_url"):
        kwargs["base_url"] = prov["base_url"]

    return LLM(**kwargs)


# ── Image helpers ─────────────────────────────────────────────────────────────

def prompt_to_query(prompt: str) -> str:
    stop = {"a", "an", "the", "of", "in", "on", "with", "and", "or", "for", "to", "at",
            "is", "are", "was", "were", "photo", "image", "illustration", "showing",
            "person", "people", "background", "surrounded", "looking", "sitting",
            "standing", "holding", "featuring", "depicting", "view", "beautiful",
            "stunning", "amazing", "great", "good", "high", "low"}
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    kw = [w for w in words if w not in stop and len(w) > 3]
    return ",".join(kw[:3]) if kw else "nature"


def get_unsplash_url(prompt: str, w: int = 1200, h: int = 630, idx: int = 0) -> str:
    return f"https://source.unsplash.com/{w}x{h}/?{prompt_to_query(prompt)}&sig={idx}"


# ── Markdown → HTML ───────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    lines, out = text.split("\n"), []
    for line in lines:
        s = line.rstrip()
        if s.startswith("### "):
            out.append(f"<h3>{s[4:]}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("# "):
            out.append(f"<h2>{s[2:]}</h2>")
        elif s == "":
            out.append("")
        else:
            s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
            out.append(f"<p>{s}</p>")
    return "\n".join(out)


# ── HTML Renderer ─────────────────────────────────────────────────────────────

def create_beautiful_html(blog: BlogPost, tone: str, day_idx: int = 0) -> str:
    accent, accent_dark = TONE_COLORS.get(tone, TONE_COLORS["Professional"])
    content_html = md_to_html(blog.content)
    tags_html = " ".join(f"<span class='tag'>#{t}</span>" for t in blog.tags)
    hero_url = get_unsplash_url(
        blog.hero_image_prompt, 1200, 630, idx=day_idx * 100)

    inline_html = ""
    for i, img in enumerate(blog.inline_images):
        img_url = get_unsplash_url(
            img.prompt, 800, 450, idx=day_idx * 100 + i + 1)
        inline_html += f"""
        <figure class="inline-img-block">
            <img src="{img_url}" alt="{img.alt_text}" loading="lazy">
            <figcaption>{img.prompt}</figcaption>
        </figure>"""

    prompts_li = "".join(
        f"<li><span class='pos-label'>{img.position}</span>{img.prompt}</li>"
        for img in blog.inline_images
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{blog.title}</title>
    <meta name="description" content="{blog.meta_description}">
    <meta name="keywords" content="{blog.focus_keyword}">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    <style>
        :root {{ --accent:{accent}; --accent-dark:{accent_dark}; --text:#1a1a1a; --muted:#6b7280; --bg:#fafaf9; --card-bg:#fff; --border:#e5e7eb; }}
        *,*::before,*::after {{ box-sizing:border-box;margin:0;padding:0; }}
        body {{ font-family:'Lora',Georgia,serif;font-weight:400;background:var(--bg);color:var(--text);line-height:1.85; }}
        #progress {{ position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent-dark));width:0%;z-index:999;transition:width .1s linear; }}
        .site-header {{ position:sticky;top:0;z-index:100;background:rgba(250,250,249,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;justify-content:space-between; }}
        .site-header .logo {{ font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:var(--accent);letter-spacing:-.02em; }}
        .site-header .day-badge {{ font-size:.78rem;color:var(--muted);background:var(--border);padding:4px 12px;border-radius:20px; }}
        .hero-wrap {{ position:relative;width:100%;max-height:520px;overflow:hidden; }}
        .hero-wrap img {{ width:100%;height:520px;object-fit:cover;display:block; }}
        .hero-overlay {{ position:absolute;inset:0;background:linear-gradient(to bottom,transparent 30%,rgba(0,0,0,.65)); }}
        .hero-text {{ position:absolute;bottom:0;left:0;right:0;padding:40px 48px;color:#fff; }}
        .hero-text h1 {{ font-family:'Playfair Display',serif;font-size:clamp(1.8rem,4.5vw,3.2rem);line-height:1.1;font-weight:900;text-shadow:0 2px 20px rgba(0,0,0,.4);margin-bottom:16px; }}
        .hero-meta {{ display:flex;gap:20px;flex-wrap:wrap;font-size:.88rem;opacity:.9; }}
        .hero-meta .pill {{ background:rgba(255,255,255,.2);backdrop-filter:blur(6px);padding:5px 14px;border-radius:20px;border:1px solid rgba(255,255,255,.3); }}
        .seo-pill {{ background:var(--accent)!important;color:#fff;font-weight:600; }}
        .tags-row {{ display:flex;flex-wrap:wrap;gap:8px;padding:24px 48px 0;max-width:860px;margin:0 auto; }}
        .tag {{ font-size:.75rem;color:var(--accent);border:1px solid var(--accent);padding:3px 12px;border-radius:20px;transition:all .2s; }}
        .tag:hover {{ background:var(--accent);color:#fff; }}
        article.body {{ max-width:720px;margin:48px auto;padding:0 24px 80px; }}
        article.body h2 {{ font-family:'Playfair Display',serif;font-size:1.65rem;font-weight:700;margin:56px 0 16px;color:var(--text);border-left:4px solid var(--accent);padding-left:16px;line-height:1.2; }}
        article.body h3 {{ font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;margin:36px 0 10px; }}
        article.body p {{ margin-bottom:1.4rem;font-size:1.05rem; }}
        article.body strong {{ font-weight:600; }}
        article.body em {{ font-style:italic;color:var(--muted); }}
        .inline-img-block {{ margin:44px -40px;border-radius:12px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.10); }}
        .inline-img-block img {{ width:100%;display:block;max-height:440px;object-fit:cover; }}
        figcaption {{ background:var(--card-bg);padding:12px 20px;font-size:.82rem;color:var(--muted);font-style:italic;border-top:1px solid var(--border); }}
        .prompts-box {{ margin-top:64px;padding:32px;background:var(--card-bg);border-radius:16px;border:1px solid var(--border);box-shadow:0 4px 16px rgba(0,0,0,.04); }}
        .prompts-box h4 {{ font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;margin-bottom:18px; }}
        .prompts-box ul {{ list-style:none;padding:0; }}
        .prompts-box li {{ padding:10px 0;font-size:.88rem;color:#555;border-bottom:1px solid var(--border);display:flex;gap:10px; }}
        .prompts-box li:last-child {{ border-bottom:none; }}
        .pos-label {{ min-width:180px;font-weight:600;color:var(--accent);font-size:.8rem;padding:2px 10px;border-radius:8px; }}
        @media(max-width:640px) {{ .hero-text{{padding:24px;}} .hero-text h1{{font-size:1.5rem;}} .tags-row{{padding:16px 16px 0;}} .inline-img-block{{margin:36px 0;}} article.body{{padding:0 16px 60px;}} }}
    </style>
</head>
<body>
<div id="progress"></div>
<header class="site-header"><span class="logo">BlogForge</span><span class="day-badge">Day {blog.day}</span></header>
<div class="hero-wrap">
    <img src="{hero_url}" alt="{blog.hero_image_prompt}">
    <div class="hero-overlay"></div>
    <div class="hero-text"><h1>{blog.title}</h1><div class="hero-meta"><span class="pill">⏱ {blog.read_time}</span><span class="pill">🔑 {blog.focus_keyword}</span><span class="pill seo-pill">SEO {blog.seo_score}/100</span></div></div>
</div>
<div class="tags-row">{tags_html}</div>
<article class="body">{content_html}{inline_html}<div class="prompts-box"><h4>📸 Image Prompts</h4><ul>{prompts_li}</ul></div></article>
<script>const bar=document.getElementById('progress');window.addEventListener('scroll',()=>{{const d=document.documentElement;bar.style.width=d.scrollTop/(d.scrollHeight-d.clientHeight)*100+'%';}});</script>
</body></html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/api/models/{provider}")
async def get_models(provider: str):
    """Return available models for a given provider."""
    prov = PROVIDER_MODELS.get(provider)
    if not prov:
        return JSONResponse(status_code=400, content={"error": f"Unknown provider: {provider}"})
    return JSONResponse(content={"provider": provider, "models": prov["models"]})


@app.post("/generate")
async def generate_blogs(request: BlogRequest):
    try:
        llm = get_llm(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
        )
        tone_instruction = TONE_INSTRUCTIONS.get(
            request.tone, TONE_INSTRUCTIONS["Professional"])

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
            verbose=True,
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

        # Get domain-specific strategy
        strategy = DOMAIN_STRATEGIES.get(domain, DOMAIN_STRATEGIES["General"])

        print(f"\n--- Detected Domain: {domain} ---")
        print(f"--- Strategy: {strategy['references']} ---")
        print(f"--- Waiting 62s to clear rate limits ---\n")
        await asyncio.sleep(62)

        # ══════════════════════════════════════════
        # PHASE 2: Write each post with domain strategy
        # ══════════════════════════════════════════
        writer = Agent(
            role="World-Class Long-Form Blogger",
            goal=f"Write comprehensive, high-quality, long-form blog posts in {request.tone} tone.",
            backstory=(
                f"You write at the level of {strategy['references']}. {tone_instruction}\n"
                f"DOMAIN STRATEGY ({domain}):\n"
                f"- Structure: {strategy['structure']}\n"
                f"- Voice: {strategy['voice']}\n"
                f"- Hooks: {strategy['hooks']}\n"
                f"- SEO approach: {strategy['seo']}\n"
                "Apply these strategies rigorously. Every post must feel like it belongs on a top-tier publication."
            ),
            verbose=True,
            llm=llm,
        )

        final_blogs = []
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, plan in enumerate(plan_data.plans):
            if i > 0:
                print(f"\n--- Waiting 15s before Day {plan.day} ---\n")
                await asyncio.sleep(15)

            write_task = Task(
                description=(
                    f"Company Context: {company_context}\n"
                    f"Domain: {domain}\n\n"
                    f"Write the blog post for DAY {plan.day}.\n"
                    f"Title: {plan.title}\n"
                    f"Focus Keyword: {plan.focus_keyword}\n"
                    f"Summary: {plan.summary}\n\n"
                    f"Tone: {request.tone} | Style: {request.style}\n\n"
                    f"DOMAIN STRATEGY TO FOLLOW ({domain} — like {strategy['references']}):\n"
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
                    "- hero_image_prompt: short noun phrase (e.g. 'apple store glass architecture night').\n"
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

            # Save files
            with open(f"reports/Day_{blog.day}_{timestamp}.md", "w", encoding="utf-8") as f:
                f.write(
                    f"# {blog.title}\n\n"
                    f"**Slug:** {blog.slug}\n"
                    f"**Focus Keyword:** {blog.focus_keyword}\n"
                    f"**Meta:** {blog.meta_description}\n"
                    f"**Read Time:** {blog.read_time} | **SEO:** {blog.seo_score}/100\n"
                    f"**Domain Strategy:** {domain} ({strategy['references']})\n"
                    f"**Tags:** {', '.join(blog.tags)}\n\n---\n\n"
                    f"{blog.content}\n\n---\n\n"
                    f"**Hero Image:** {blog.hero_image_prompt}\n"
                )

            with open(f"reports/Day_{blog.day}_{timestamp}.html", "w", encoding="utf-8") as f:
                f.write(create_beautiful_html(
                    blog, request.tone, blog.day - 1))

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
