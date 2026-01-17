import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# =========================
# PAGE CONFIG (FIRST!)
# =========================
st.set_page_config(
    page_title="NEO — Диагностика по позициям (AI-only)",
    page_icon="💠",
    layout="centered",
)

# =========================
# CONFIG LOAD
# =========================
def load_config():
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))

CFG = load_config()
APP_TITLE = CFG.get("app", {}).get("title", "💠 NEO — Диагностика по позициям (AI-only)")
APP_VERSION = CFG.get("app", {}).get("version", "positions-ai-1.0")

DATA_DIR = Path(CFG.get("storage", {}).get("data_dir", "data/sessions"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = CFG.get("openai", {}).get("model", "gpt-4.1-mini")

MASTER_PASSWORD_ENV = CFG.get("master", {}).get("password_env", "MASTER_PASSWORD")
MASTER_PASSWORD = st.secrets.get("MASTER_PASSWORD", os.getenv(MASTER_PASSWORD_ENV, ""))

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

MAX_Q_TOTAL = int(CFG.get("flow", {}).get("max_questions_total", 24))
MAX_FOLLOWUPS_PER_STEP = int(CFG.get("flow", {}).get("max_followups_per_step", 1))
CONF_STOP = float(CFG.get("flow", {}).get("confidence_stop", 0.78))

# =========================
# KNOWLEDGE LOAD
# =========================
def load_positions_knowledge() -> str:
    p = Path("knowledge/positions.md")
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8")
    # лёгкий safety-trim (чтобы не улетать в огромные токены)
    return txt[:22000]

POSITIONS_KNOWLEDGE = load_positions_knowledge()

# =========================
# OPENAI
# =========================
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

def safe_model_name(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return DEFAULT_MODEL
    # чтобы не словить 404 на gpt-5.x
    if m.startswith("gpt-5"):
        return DEFAULT_MODEL
    return m

# =========================
# STORAGE
# =========================
def utcnow_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def session_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"

def save_session(payload: dict):
    session_path(payload["meta"]["session_id"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def load_session(session_id: str):
    p = session_path(session_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def list_sessions():
    items = []
    for p in sorted(DATA_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items

# =========================
# STATE MACHINE (AI-ONLY)
# =========================
POTS = ["Сапфир","Гелиодор","Аметист","Изумруд","Гранат","Рубин","Янтарь","Шунгит","Цитрин"]

STEPS = [
    # позиция 1: 2 вопроса сфера + 2 вопроса уточнение потенциала
    {"id":"p1_scope_1","title":"Позиция 1 — сфера (1/2)","goal":"определить сферу/тип восприятия позиции 1"},
    {"id":"p1_scope_2","title":"Позиция 1 — сфера (2/2)","goal":"дожать сферу позиции 1"},
    {"id":"p1_pot_1","title":"Позиция 1 — потенциал (1/2)","goal":"сузить до конкретного потенциала позиции 1"},
    {"id":"p1_pot_2","title":"Позиция 1 — потенциал (2/2)","goal":"зафиксировать потенциал позиции 1"},

    # позиция 2
    {"id":"p2_scope_1","title":"Позиция 2 — сфера (1/2)","goal":"определить сферу позиции 2"},
    {"id":"p2_scope_2","title":"Позиция 2 — сфера (2/2)","goal":"дожать сферу позиции 2"},
    {"id":"p2_pot_1","title":"Позиция 2 — потенциал (1/2)","goal":"сузить до конкретного потенциала позиции 2"},
    {"id":"p2_pot_2","title":"Позиция 2 — потенциал (2/2)","goal":"зафиксировать потенциал позиции 2"},

    # позиция 3
    {"id":"p3_scope_1","title":"Позиция 3 — сфера (1/2)","goal":"определить сферу позиции 3"},
    {"id":"p3_scope_2","title":"Позиция 3 — сфера (2/2)","goal":"дожать сферу позиции 3"},
    {"id":"p3_pot_1","title":"Позиция 3 — потенциал (1/2)","goal":"сузить до конкретного потенциала позиции 3"},
    {"id":"p3_pot_2","title":"Позиция 3 — потенциал (2/2)","goal":"зафиксировать потенциал позиции 3"},
]

# =========================
# SESSION STATE
# =========================
def init_state():
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("client_name", "")
    st.session_state.setdefault("client_contact", "")
    st.session_state.setdefault("client_request", "")

    st.session_state.setdefault("step_index", 0)
    st.session_state.setdefault("q_count", 0)

    # диалог: список {role, content}
    st.session_state.setdefault("messages", [])
    # ответы: список {step_id, question, answer}
    st.session_state.setdefault("answers", [])

    # текущий вопрос (AI)
    st.session_state.setdefault("current_q", None)  # dict
    st.session_state.setdefault("current_answer", "")  # text

    # результаты
    st.session_state.setdefault("positions", {"p1":None,"p2":None,"p3":None})
    st.session_state.setdefault("confidence", {"p1":0.0,"p2":0.0,"p3":0.0})
    st.session_state.setdefault("scores", {p:0.0 for p in POTS})
    st.session_state.setdefault("col_scores", {
        "perception": {p:0.0 for p in POTS},
        "motivation": {p:0.0 for p in POTS},
        "tool": {p:0.0 for p in POTS},
        "result": {p:0.0 for p in POTS},
    })

    st.session_state.setdefault("done", False)

    # master auth
    st.session_state.setdefault("master_authed", False)

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()

# =========================
# AI CONTRACT
# =========================
def build_system_prompt():
    return f"""
Ты — интервьюер по авторской методике "Потенциалы". У тебя есть база знаний ниже.
Твоя задача: вести интервью СТРОГО по шагам и выдавать вопрос + варианты (если подходят).

ЖЁСТКИЕ ПРАВИЛА:
1) Не задавай больше 1 уточняющего вопроса на шаг.
2) Не "мусоль" эмоции. Один уточняющий максимум.
3) Каждый вопрос должен быть бытовым, понятным, с конкретной ситуацией.
4) Если возможно, давай 4 варианта ответа (короткие). Всегда добавляй вариант "Другое (своими словами)".
5) Ты НЕ раскрываешь названия потенциалов клиенту (камни) в клиентском режиме.
6) Ты всегда возвращаешь строго JSON по схеме.

БАЗА ЗНАНИЙ (сокращение, используй как источник):
---BEGIN KNOWLEDGE---
{POSITIONS_KNOWLEDGE}
---END KNOWLEDGE---
""".strip()

def build_user_payload(step_id: str, step_goal: str):
    return {
        "mode": "client_interview",
        "step_id": step_id,
        "step_goal": step_goal,
        "client": {
            "name": st.session_state.get("client_name",""),
            "contact": st.session_state.get("client_contact",""),
            "request": st.session_state.get("client_request",""),
        },
        "state": {
            "step_index": st.session_state["step_index"],
            "q_count": st.session_state["q_count"],
            "positions": st.session_state["positions"],
            "confidence": st.session_state["confidence"],
            "scores": st.session_state["scores"],
            "col_scores": st.session_state["col_scores"],
        },
        "history_tail": st.session_state["answers"][-6:],  # последние ответы
        "limits": {
            "max_questions_total": MAX_Q_TOTAL,
            "confidence_stop": CONF_STOP
        }
    }

def response_schema_note():
    return {
        "type": "json_object"
    }

# ожидаемый JSON от модели:
# {
#  "question": "...",
#  "type": "single"|"text",
#  "options": ["..."] (если single),
#  "analysis_update": {
#    "scores_delta": {"Аметист":0.2, ...},
#    "col_scores_delta": {"perception":{"...":0.1}, "motivation":{...}, "tool":{...}, "result":{...}},
#    "positions_guess": {"p1":"...", "p2":"...", "p3":"..."} (может быть null),
#    "confidence": {"p1":0.0-1.0, ...},
#    "notes_for_master": "..."
#  }
# }

def call_ai_next_question(client, model: str, step_id: str, step_goal: str):
    sys = build_system_prompt()
    payload = build_user_payload(step_id, step_goal)

    guide = """
Верни JSON:
- question (строка)
- type: "single" или "text"
- options: если type="single", дай 4-5 вариантов, последний обязательно "Другое (своими словами)"
- analysis_update: объект, где:
    - scores_delta: словарь по камням (можно частично)
    - col_scores_delta: словарь по колонкам (perception/motivation/tool/result) → частично
    - positions_guess: p1/p2/p3 (можно null)
    - confidence: p1/p2/p3 (0..1)
    - notes_for_master: коротко
""".strip()

    r = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":sys},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)},
            {"role":"user","content": guide},
        ],
        response_format=response_schema_note(),
    )

    text = ""
    try:
        text = r.output_text
    except Exception:
        # fallback
        text = json.dumps({"question":"Ошибка чтения ответа модели","type":"text","options":[],"analysis_update":{}}, ensure_ascii=False)

    return json.loads(text)

def apply_analysis_update(update: dict):
    if not isinstance(update, dict):
        return

    # scores
    d = update.get("scores_delta", {})
    if isinstance(d, dict):
        for k, v in d.items():
            if k in st.session_state["scores"]:
                st.session_state["scores"][k] = float(st.session_state["scores"][k]) + float(v)

    # col scores
    cd = update.get("col_scores_delta", {})
    if isinstance(cd, dict):
        for col in ["perception","motivation","tool","result"]:
            sub = cd.get(col, {})
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if k in st.session_state["col_scores"][col]:
                        st.session_state["col_scores"][col][k] = float(st.session_state["col_scores"][col][k]) + float(v)

    # guesses
    g = update.get("positions_guess", {})
    if isinstance(g, dict):
        for p in ["p1","p2","p3"]:
            if p in g and g[p]:
                st.session_state["positions"][p] = g[p]

    # confidence
    c = update.get("confidence", {})
    if isinstance(c, dict):
        for p in ["p1","p2","p3"]:
            if p in c and c[p] is not None:
                st.session_state["confidence"][p] = float(c[p])

def topn(d: dict, n=3):
    return sorted(d.items(), key=lambda x: float(x[1]), reverse=True)[:n]

# =========================
# REPORTS
# =========================
def build_client_report_text():
    # без названий камней
    vecs = []
    # просто по топам в колонках (без камней)
    for col, label in [
        ("perception","Как вы воспринимаете мир"),
        ("motivation","Что вас реально мотивирует"),
        ("tool","Какой инструмент включается"),
        ("result","Какой результат вы обычно даёте"),
    ]:
        top = topn(st.session_state["col_scores"][col], 2)
        # мы не показываем названия, только описания “вектора”
        # пока делаем мягко: "есть выраженные сигналы X/Y"
        vecs.append(f"- **{label}**: есть выраженные сигналы по 2 направлениям (уточняется на встрече).")

    name = st.session_state.get("client_name","")
    req = st.session_state.get("client_request","")

    lines = [
        f"**{name}**, вот ваш **предварительный отчёт** по диагностике (AI-only).",
        "",
        f"**Запрос:** {req}",
        "",
        "**Что уже видно:**",
        "1) У вас есть сильный ресурс в стратегировании/управлении и движении к результату.",
        "2) Реализация быстрее приходит, когда вы *проявляетесь* (голос/эмоция/контакт с людьми), а не “делаете в тишине и в одиночку”.",
        "3) Сливы энергии обычно появляются там, где много рутины, регламентов или “делаю — но не чувствую смысла”.",
        "",
        "**4 колонки (векторно):**",
        *vecs,
        "",
        "**Почему важно не тянуть:**",
        "Если вы долго подавляете свою природную манеру проявления, появляется ощущение усталости, пустоты и “я не там”.",
        "",
        "**Следующий шаг:**",
        "Чтобы зафиксировать потенциалы *по позициям* (и точно разложить реализацию/деньги/проявленность), рекомендую встречу с мастером: там мы уточним спорные места и дадим полный отчёт + план.",
    ]
    return "\n".join(lines)

def call_ai_master_report(client, model: str, payload: dict):
    sys = (
        "Ты мастер-диагност. Дай МАСТЕР-ОТЧЁТ строго структурно:\n"
        "1) Таблица позиций: P1/P2/P3 (потенциал + краткий маркер)\n"
        "2) Колонки: perception/motivation/tool/result: топ-2 потенциала и почему\n"
        "3) Конфликты/смещения\n"
        "4) 6 уточняющих вопросов (короткие)\n"
        "5) Рекомендации по реализации/монетизации под запрос\n"
        "Пиши по-русски, конкретно."
    )

    r = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":sys},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ],
    )
    try:
        return r.output_text
    except Exception:
        return "Не удалось прочитать ответ модели."

def build_payload_final():
    ranked = sorted(st.session_state["scores"].items(), key=lambda x: float(x[1]), reverse=True)
    payload = {
        "meta": {
            "schema": "ai-neo.positions.ai_only.v1",
            "app_version": APP_VERSION,
            "timestamp": utcnow_iso(),
            "session_id": st.session_state["session_id"],
            "name": st.session_state.get("client_name",""),
            "contact": st.session_state.get("client_contact",""),
            "request": st.session_state.get("client_request",""),
            "q_count": st.session_state.get("q_count", 0),
            "model": st.session_state.get("model_used", DEFAULT_MODEL),
        },
        "answers": st.session_state["answers"],
        "positions_guess": st.session_state["positions"],
        "confidence": st.session_state["confidence"],
        "scores": st.session_state["scores"],
        "col_scores": st.session_state["col_scores"],
        "top6": [{"pot":p,"score":float(s)} for p,s in ranked[:6]],
    }
    return payload

# =========================
# UI: CLIENT FLOW
# =========================
def client_intake():
    st.markdown("### Старт")
    st.caption("Отвечай быстро и честно. Не выбирай «как правильно», выбирай «как у меня».")

    st.session_state["client_name"] = st.text_input("Как к тебе обращаться?", value=st.session_state.get("client_name",""), key="in_name")
    st.session_state["client_contact"] = st.text_input("Телефон или email (куда отправить полный отчёт)", value=st.session_state.get("client_contact",""), key="in_contact")
    st.session_state["client_request"] = st.text_input("С каким запросом ты пришёл(пришла)? (1 фраза)", value=st.session_state.get("client_request",""), key="in_req")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🚀 Начать диагностику", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["answers"] = []
            st.session_state["step_index"] = 0
            st.session_state["q_count"] = 0
            st.session_state["done"] = False
            st.session_state["current_q"] = None
            st.session_state["current_answer"] = ""
            st.rerun()
    with c2:
        if st.button("🔄 Полный сброс", use_container_width=True):
            reset_all()
            st.rerun()

def ensure_current_question():
    if st.session_state["done"]:
        return

    if st.session_state["q_count"] >= MAX_Q_TOTAL:
        st.session_state["done"] = True
        return

    # стоп по уверенностям (если все 3 позиции уже уверенно)
    c = st.session_state["confidence"]
    if c.get("p1",0) >= CONF_STOP and c.get("p2",0) >= CONF_STOP and c.get("p3",0) >= CONF_STOP:
        st.session_state["done"] = True
        return

    if st.session_state["current_q"] is not None:
        return

    # получить следующий шаг
    idx = st.session_state["step_index"]
    if idx >= len(STEPS):
        st.session_state["done"] = True
        return

    client = get_openai_client()
    if not client:
        st.error("Нет OPENAI_API_KEY. Добавь в Streamlit secrets или env.")
        st.stop()

    model = safe_model_name(st.session_state.get("model_used", DEFAULT_MODEL))
    st.session_state["model_used"] = model

    step = STEPS[idx]
    qjson = call_ai_next_question(client, model, step["id"], step["goal"])
    st.session_state["current_q"] = qjson
    st.session_state["current_answer"] = ""

def render_current_question():
    q = st.session_state.get("current_q")
    if not q:
        return

    idx = st.session_state["step_index"]
    step_title = STEPS[idx]["title"] if idx < len(STEPS) else "—"

    st.markdown(f"### {step_title}")
    st.markdown(q.get("question","(вопрос отсутствует)"))

    qtype = q.get("type","text")
    options = q.get("options", [])

    # ключ уникальный: вопрос + session_id + step_index => текст не переносится
    ui_key = f"ans_{st.session_state['session_id']}_{idx}"

    ans = None
    if qtype == "single" and isinstance(options, list) and len(options) > 0:
        ans = st.radio("Выбери вариант:", options, key=ui_key)
        # если выбрали "другое" — появится поле
        if isinstance(ans, str) and ans.lower().startswith("другое"):
            free = st.text_area("Напиши своими словами:", key=f"{ui_key}_free", height=120)
            if free.strip():
                ans = free.strip()
    else:
        ans = st.text_area("Ответ:", key=ui_key, height=150)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Далее ➜", use_container_width=True):
            if not str(ans or "").strip():
                st.warning("Заполни ответ.")
                return

            # сохранить ответ
            st.session_state["answers"].append({
                "step_id": STEPS[st.session_state["step_index"]]["id"] if st.session_state["step_index"] < len(STEPS) else "done",
                "question": q.get("question",""),
                "answer": ans,
                "timestamp": utcnow_iso()
            })

            # применить апдейт скоринга
            apply_analysis_update(q.get("analysis_update", {}))

            # двигаться дальше
            st.session_state["q_count"] += 1
            st.session_state["step_index"] += 1
            st.session_state["current_q"] = None  # важно: сброс, чтобы получить новый вопрос

            st.rerun()

    with c2:
        if st.button("Завершить сейчас", use_container_width=True):
            st.session_state["done"] = True
            st.session_state["current_q"] = None
            st.rerun()

def render_done():
    st.success("Диагностика завершена ✅")

    payload = build_payload_final()
    save_session(payload)

    st.markdown("## Мини-отчёт (для клиента)")
    st.markdown(build_client_report_text())

    with st.expander("Показать мои ответы (для проверки)"):
        st.json(payload.get("answers", []))

# =========================
# MASTER PANEL
# =========================
def render_master_panel():
    st.subheader("🛠️ Мастер-панель")

    if not MASTER_PASSWORD:
        st.warning("MASTER_PASSWORD не задан. Задай его в secrets/env.")
        return

    if not st.session_state.get("master_authed", False):
        pwd = st.text_input("Пароль мастера", type="password", key="mpwd")
        if st.button("Войти", use_container_width=True):
            if pwd == MASTER_PASSWORD:
                st.session_state["master_authed"] = True
                st.success("Ок ✅")
                st.rerun()
            else:
                st.error("Неверный пароль")
        st.stop()

    sessions = list_sessions()
    if not sessions:
        st.info("Пока нет сохранённых сессий.")
        st.stop()

    labels, ids = [], []
    for s in sessions:
        meta = s.get("meta", {})
        sid = meta.get("session_id", "")
        labels.append(f"{meta.get('name','—')} | {meta.get('request','—')} | {meta.get('timestamp','—')} | {sid[:8]}")
        ids.append(sid)

    pick = st.selectbox("Сессии:", labels, index=0, key="pick")
    chosen_id = ids[labels.index(pick)]
    payload = load_session(chosen_id)

    if not payload:
        st.error("Не удалось загрузить сессию.")
        st.stop()

    meta = payload.get("meta", {})
    st.markdown(
        f"**Имя:** {meta.get('name','—')}\n\n"
        f"**Контакт:** {meta.get('contact','—')}\n\n"
        f"**Запрос:** {meta.get('request','—')}\n\n"
        f"**Вопросов:** {meta.get('q_count','—')}\n"
    )

    st.download_button(
        "⬇️ Скачать JSON (сессия)",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"session_{chosen_id[:8]}.json",
        mime="application/json",
        use_container_width=True
    )

    with st.expander("📌 Таблица (для мастера)"):
        st.json({
            "positions_guess": payload.get("positions_guess"),
            "confidence": payload.get("confidence"),
            "top6": payload.get("top6"),
            "col_scores_top2": {
                col: topn(payload.get("col_scores", {}).get(col, {}), 2)
                for col in ["perception","motivation","tool","result"]
            }
        })

    st.markdown("---")
    st.subheader("🧠 Мастерский AI-отчёт")

    model_in = st.text_input("Модель", value=DEFAULT_MODEL, key="mmodel")

    if st.button("Сгенерировать мастер-отчёт", use_container_width=True):
        client = get_openai_client()
        if not client:
            st.error("Нет OPENAI_API_KEY")
        else:
            model = safe_model_name(model_in)
            report = call_ai_master_report(client, model, payload)
            payload["ai_master_report"] = report
            save_session(payload)
            st.success("Готово ✅ Сохранено в сессии.")
            st.write(report)

    if payload.get("ai_master_report"):
        with st.expander("Показать сохранённый мастер-отчёт"):
            st.write(payload["ai_master_report"])

# =========================
# MAIN
# =========================
init_state()

st.title(APP_TITLE)
st.caption(f"Версия: {APP_VERSION}")

tab1, tab2 = st.tabs(["🧑‍💼 Клиент", "🛠️ Мастер"])

with tab1:
    # Intake если еще не начали
    if not st.session_state["answers"] and not st.session_state["done"] and st.session_state["current_q"] is None:
        client_intake()

    if not st.session_state["done"]:
        ensure_current_question()
        if not st.session_state["done"]:
            st.caption(f"Прогресс: {st.session_state['q_count']} / {MAX_Q_TOTAL}")
            render_current_question()
        else:
            render_done()
    else:
        render_done()

with tab2:
    render_master_panel()
