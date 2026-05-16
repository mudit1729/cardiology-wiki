from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path

import requests
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    stream_with_context,
    url_for,
)

from wiki_loader import WikiRepository


BASE_DIR = Path(__file__).resolve().parent

# Load local .env values for the paper-ingest pipeline regardless of how Flask is started.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates_html"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("WIKI_SECRET_KEY") or secrets.token_hex(32)

_WIKI_PASSWORD = os.environ.get("WIKI_PASSWORD", "Asd1729@")

_PUBLIC_ENDPOINTS = {"login", "static"}

repo = WikiRepository(BASE_DIR)

CHAT_TOPICS = [
    ("all", "All Pages"),
    ("stable-cad", "Stable CAD"),
    ("acs", "ACS / STEMI"),
    ("antiplatelet", "Antiplatelets"),
    ("ivus", "IVUS / OCT"),
    ("ffr", "FFR / iFR"),
    ("tavr", "TAVR"),
    ("heart-failure", "Heart Failure"),
    ("lipid", "Lipids / Prevention"),
    ("atrial-fibrillation", "AF + PCI"),
    ("pci", "PCI / Procedures"),
    ("guideline", "Guidelines"),
]

CHAT_CONTEXT_CHAR_LIMIT = int(os.environ.get("CHAT_CONTEXT_CHAR_LIMIT", "60000"))
CHAT_MAX_PAPERS = int(os.environ.get("CHAT_MAX_PAPERS", "40"))


@app.before_request
def _require_login():
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = request.values.get("next") or url_for("home")
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = url_for("home")
    if request.method == "POST":
        submitted = request.form.get("password", "")
        if hmac.compare_digest(submitted, _WIKI_PASSWORD):
            session.clear()
            session["authed"] = True
            session.permanent = True
            return redirect(next_url)
        error = "Incorrect password."
    return render_template("login.html", error=error, next_url=next_url), (401 if error else 200)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.context_processor
def inject_globals() -> dict[str, object]:
    return {
        "repo_title": "Cardio Wiki",
        "repo_title_long": "Interventional cardiology intelligence — trials, guidelines, procedures, and practice-changing updates",
    }


@app.route("/")
def home():
    featured = repo.featured_pages()
    groups = repo.grouped_pages()
    recent = repo.recent_log_entries()
    stats = repo.stats()
    collections = repo.paper_collections()
    return render_template(
        "home.html",
        featured=featured,
        groups=groups,
        recent=recent,
        stats=stats,
        collections=collections,
    )


@app.route("/page/<path:page_path>")
def page(page_path: str):
    page_obj = repo.get_page(page_path)
    if not page_obj:
        abort(404)

    breadcrumbs = []
    parts = page_obj.path.split("/")
    for index in range(len(parts)):
        crumb_path = "/".join(parts[: index + 1])
        breadcrumbs.append(
            {
                "label": parts[index].replace("-", " ").title(),
                "path": crumb_path,
                "exists": bool(repo.get_page(crumb_path)),
            }
        )

    backlinks = [repo.get_page(path) for path in page_obj.backlinks]
    backlinks = [page for page in backlinks if page]

    return render_template(
        "page.html",
        page=page_obj,
        breadcrumbs=breadcrumbs,
        backlinks=backlinks,
    )


@app.route("/papers")
def papers():
    all_papers = repo.papers()
    collections = repo.paper_collections()
    tag_filter = request.args.get("tag", "").strip()
    if tag_filter:
        all_papers = [p for p in all_papers if tag_filter in p.tags]
    return render_template(
        "papers.html",
        papers=all_papers,
        collections=collections,
        active_tag=tag_filter,
    )


@app.route("/graph")
def graph():
    graph_data = repo.paper_graph_data()
    return render_template(
        "graph.html",
        graph_json=json.dumps(graph_data),
    )


@app.route("/timeline")
def timeline():
    timeline_data = repo.timeline_data()
    return render_template(
        "timeline.html",
        timeline_data=timeline_data,
        timeline_json=json.dumps(timeline_data),
    )


@app.route("/tags")
def tags():
    all_tags = repo.all_tags()
    tag_name = request.args.get("t", "").strip()
    tag_pages = repo.pages_by_tag(tag_name) if tag_name else []
    return render_template(
        "tags.html",
        all_tags=all_tags,
        active_tag=tag_name,
        tag_pages=tag_pages,
    )


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = repo.search(query) if query else []
    return render_template("search.html", query=query, results=results)


@app.route("/vault/<path:file_path>")
def vault_file(file_path: str):
    target = (BASE_DIR / file_path).resolve()
    raw_dir = (BASE_DIR / "raw").resolve()
    if raw_dir not in target.parents and target != raw_dir:
        abort(404)
    if not target.exists():
        abort(404)
    return send_from_directory(target.parent, target.name)


@app.route("/add-paper")
def add_paper_page():
    return render_template("add-paper.html")


@app.route("/api/papers/related")
def papers_related():
    group = (request.args.get("group") or "").strip().lower()
    graph = repo.paper_graph_data()
    nodes = graph.get("nodes") or []

    group_tags = {
        "stable-cad": {"stable-cad", "chronic-coronary-disease"},
        "acs": {"acs", "stemi", "nstemi", "acute-coronary-syndrome"},
        "imaging-guided-pci": {"ivus", "oct", "imaging-guided-pci", "intravascular-imaging"},
        "physiology-guided-pci": {"ffr", "ifr", "physiology-guided-pci", "coronary-physiology"},
        "antiplatelet": {"antiplatelet", "dapt", "dual-antiplatelet", "antithrombotic"},
        "af-pci": {"atrial-fibrillation", "af-pci", "anticoagulation"},
        "structural-heart": {"tavr", "structural-heart", "aortic-stenosis", "tavi", "mitral"},
        "lipid-prevention": {"lipid", "lipid-prevention", "pcsk9", "statin", "prevention", "glp1"},
        "heart-failure": {"heart-failure", "sglt2", "hfref", "hfpef"},
    }
    wanted = group_tags.get(group)
    if wanted:
        nodes = [n for n in nodes if wanted & set(n.get("tags") or [])]
    elif group and group not in {"all", ""}:
        nodes = []

    def _int_value(value):
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    nodes = sorted(nodes, key=lambda n: (-_int_value(n.get("citations")), -_int_value(n.get("year"))))
    papers = [
        {
            "path": n["id"],
            "title": n.get("title"),
            "year": n.get("year"),
            "group": n.get("group"),
            "citations": n.get("citations"),
            "tags": n.get("tags") or [],
        }
        for n in nodes[:30]
    ]
    return jsonify({"group": group or "all", "count": len(nodes), "papers": papers})


@app.route("/api/papers/ingest", methods=["POST"])
def papers_ingest():
    from paper_ingest import ingest_pipeline

    url = (request.form.get("url") or "").strip()
    title_hint = (request.form.get("title") or "").strip() or None
    slug_hint = (request.form.get("slug") or "").strip() or None

    pdf_bytes = None
    f = request.files.get("pdf")
    if f and f.filename:
        pdf_bytes = f.read()
        if len(pdf_bytes) < 1000:
            return jsonify({"error": "PDF too small / empty"}), 400

    if not url and not pdf_bytes:
        return jsonify({"error": "Provide either a URL or upload a PDF"}), 400

    def _flag(name: str, default: bool = True) -> bool:
        value = request.form.get(name)
        if value is None:
            return default
        return value not in {"0", "false", "no", "off", ""}

    group_hint = (request.form.get("group") or "").strip().lower() or None
    do_citation = _flag("do_citation")
    do_openai = _flag("do_openai")
    do_autopush = _flag("do_autopush")

    def stream():
        for name, payload in ingest_pipeline(
            url=url or None,
            pdf_bytes=pdf_bytes,
            title_hint=title_hint,
            slug_hint=slug_hint,
            source_url=url or None,
            group_hint=group_hint,
            do_citation=do_citation,
            do_openai=do_openai,
            do_autopush=do_autopush,
        ):
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/chat")
def chat():
    papers_for_chat = []
    for paper in repo.papers():
        papers_for_chat.append(
            {
                "path": paper.path,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "citations": paper.citations,
                "tags": paper.tags,
                "excerpt": paper.excerpt,
            }
        )
    return render_template(
        "chat.html",
        papers=papers_for_chat,
        topics=CHAT_TOPICS,
        max_papers=CHAT_MAX_PAPERS,
    )


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        return jsonify({"error": "DEEPSEEK_API_KEY is not configured"}), 503

    selected_paths = payload.get("paper_paths") or []
    if not isinstance(selected_paths, list):
        return jsonify({"error": "paper_paths must be a list"}), 400
    selected_paths = [str(path) for path in selected_paths[:CHAT_MAX_PAPERS]]

    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    history = _clean_chat_history(history)

    thinking_mode = str(payload.get("thinking_mode", "low")).lower()
    if thinking_mode not in {"low", "high"}:
        thinking_mode = "low"

    context_text, included_papers = _build_chat_context(selected_paths)
    messages = _build_deepseek_messages(user_message, history, context_text, included_papers)
    deepseek_payload = _build_deepseek_payload(messages, thinking_mode)

    return Response(
        stream_with_context(_stream_deepseek(deepseek_payload, api_key, included_papers)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _clean_chat_history(history: list[object], limit: int = 10) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:6000]})
    return cleaned


def _strip_frontmatter(markdown_text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", markdown_text, count=1, flags=re.DOTALL).strip()


def _build_chat_context(paths: list[str]) -> tuple[str, list[dict[str, str]]]:
    chunks: list[str] = []
    included: list[dict[str, str]] = []
    remaining = CHAT_CONTEXT_CHAR_LIMIT

    for path in paths:
        page = repo.get_page(path)
        if not page:
            continue
        body = _strip_frontmatter(page.raw)
        header = (
            f"## {page.title}\n"
            f"Link: [{page.title}](/page/{page.path})\n"
            f"Type: {page.page_type or 'unknown'}\n"
            f"Year: {page.year or 'unknown'}\n"
            f"Tags: {', '.join(page.tags) if page.tags else 'none'}\n\n"
        )
        per_paper_budget = max(2000, min(9000, remaining // max(1, len(paths) - len(included))))
        summary = body[: max(0, per_paper_budget - len(header))]
        if len(body) > len(summary):
            summary = summary.rstrip() + "\n\n[Summary truncated for context budget.]"
        chunk = header + summary
        if len(chunk) > remaining:
            break
        chunks.append(chunk)
        included.append({"title": page.title, "path": page.path})
        remaining -= len(chunk)
        if remaining <= 2000:
            break

    return "\n\n---\n\n".join(chunks), included


def _build_deepseek_messages(
    user_message: str,
    history: list[dict[str, str]],
    context_text: str,
    included_papers: list[dict[str, str]],
) -> list[dict[str, str]]:
    paper_list = "\n".join(
        f"- [{paper['title']}](/page/{paper['path']}) — wiki path: /page/{paper['path']}"
        for paper in included_papers
    )
    system = (
        "You are the research chat assistant for the Cardio Wiki — an interventional cardiology "
        "intelligence system covering practice-changing trials, guidelines, PCI techniques, "
        "structural heart interventions, and India-specific practice considerations. "
        "Topics include stable CAD, ACS/STEMI, complex PCI, CTO, imaging-guided PCI (IVUS/OCT), "
        "physiology-guided PCI (FFR/iFR), antiplatelet therapy, TAVR, heart failure, and lipid management. "
        "Answer using the selected page summaries as your primary evidence. "
        "When citing a trial or guideline, ALWAYS use a markdown link with the wiki path: "
        "[Trial Name](/page/wiki/trials/domain/trial-slug). "
        "The paths for each page are provided in the context. "
        "Separate direct source claims from synthesis when the distinction matters. "
        "If the selected pages do not contain enough evidence, say what is missing. "
        "Always note India-specific practice implications when relevant. "
        "Do not invent trial results, patient numbers, or statistical values."
    )
    context_message = (
        "Selected page context:\n"
        f"{paper_list or '- No pages were selected.'}\n\n"
        f"{context_text or '[No selected page context.]'}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context_message},
        *history,
        {"role": "user", "content": user_message},
    ]


def _build_deepseek_payload(messages: list[dict[str, str]], thinking_mode: str) -> dict:
    model = (
        os.environ.get("DEEPSEEK_HIGH_MODEL", "deepseek-v4-pro")
        if thinking_mode == "high"
        else os.environ.get("DEEPSEEK_LOW_MODEL", "deepseek-v4-flash")
    )
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled" if thinking_mode == "high" else "disabled"},
    }
    if thinking_mode == "high":
        payload["reasoning_effort"] = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
    else:
        payload["temperature"] = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.2"))
    return payload


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_deepseek(payload: dict, api_key: str, included_papers: list[dict[str, str]]):
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    yield _sse(
        "meta",
        {
            "model": payload["model"],
            "paper_count": len(included_papers),
            "papers": included_papers,
        },
    )

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 180)) as response:
            if response.status_code >= 400:
                yield _sse("error", {"message": response.text[:1000] or response.reason})
                return

            thinking_started = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    yield _sse("done", {})
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    usage = chunk.get("usage")
                    if usage:
                        yield _sse("usage", usage)
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content") and not thinking_started:
                    thinking_started = True
                    yield _sse("status", {"message": "Thinking..."})
                content = delta.get("content") or ""
                if content:
                    yield _sse("token", {"text": content})
            yield _sse("done", {})
    except requests.RequestException as exc:
        yield _sse("error", {"message": str(exc)})


from update_citations import register_citation_routes, run_update

register_citation_routes(app)


_citation_status = {
    "last_run": None,
    "last_result": None,
    "running": False,
}


def _run_citation_update_background():
    _citation_status["running"] = True
    try:
        result = run_update(dry_run=False, stale_days=7)
        _citation_status["last_result"] = result
    except Exception as e:
        _citation_status["last_result"] = {"error": str(e)}
    finally:
        _citation_status["last_run"] = datetime.now().isoformat()
        _citation_status["running"] = False


@app.route("/api/citations/status")
def citation_status():
    if not session.get("authed"):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(_citation_status)


@app.route("/api/citations/trigger", methods=["POST"])
def citation_trigger():
    if not session.get("authed"):
        return jsonify({"error": "unauthorized"}), 401
    if _citation_status["running"]:
        return jsonify({"error": "already running"}), 409
    thread = threading.Thread(target=_run_citation_update_background, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


def _start_scheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _run_citation_update_background,
            "interval",
            days=7,
            id="citation_update_weekly",
            replace_existing=True,
        )
        scheduler.start()
        print("[scheduler] Citation update scheduled every 7 days")
    except ImportError:
        print("[scheduler] APScheduler not installed, skipping auto-scheduling")
    except Exception as e:
        print(f"[scheduler] Failed to start: {e}")


if not os.environ.get("WERKZEUG_RUN_MAIN") and not os.environ.get("FLASK_DEBUG"):
    _start_scheduler()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8081")), debug=True)
