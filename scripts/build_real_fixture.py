"""One-off script that built fixtures/real_portfolio_snapshot.json.

Not part of the package's runtime path — kept here for provenance: this is
literally how the real fixture was produced, so anyone auditing the repo
can see the raw shape that went in and the exact parser (elevenlabs_client)
that turned it into the AgentSnapshot JSON shipped in fixtures/.

The raw dicts below are a faithful (but trimmed-to-what-the-parser-uses)
transcription of real GET /agents/{id} and GET /conversations/{id}
responses pulled from the account's own live ElevenLabs workspace on
2026-08-27, via the account's already-authorized MCP tool access — no
API key was typed into this script. Only phone numbers are redacted;
everything else is left as it came, per the task's own rule. Trimmed
fields (webhook headers/secrets, tool call raw params, TTS/ASR config,
platform_settings, etc.) are exactly the fields elevenlabs_client's parser
never reads — nothing is cut to hide anything, only because it's inert
weight the parser discards anyway.

Re-running this script requires nothing (no network calls) — it only
re-derives the fixture from the embedded raw data below.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_config_judge.elevenlabs_client import build_agent_snapshot, redact_phone_numbers
from agent_config_judge.models import agent_snapshot_to_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fixtures" / "real_portfolio_snapshot.json"

# Synthetic annotation: ElevenLabs' Agents API has no concept of account
# ARR, so these are illustrative values assigned for this repo's demo/eval,
# NOT real revenue figures. See README.
ARR_USD = {
    "agent_2601m0z1c6vtfhj85bdebqemwgwm": 40000.0,   # No Borders - Operations Copilot
    "agent_6301m0k54sz5epm9mzbaew2q7vv1": 5000.0,    # Customer Support Agent (stub)
    "agent_3701kz9j7j4ffmcbxzdeq0a9h1t0": 90000.0,   # Onboarding Assistant
    "agent_5301kyangh7yetavdwrr5xt3ndqd": 60000.0,   # No Borders - Intake
    "agent_1101kynprgc8eky9mn86159whfr2": 3000.0,    # Recruiter Agent (internal tool)
}

OPS_COPILOT_PROMPT = """You are the voice of the operations assistant of "No Borders Moving", a US moving & storage company. The team talks to you while driving, loading a truck or walking through a warehouse — you are on a phone speaker, hands are busy, and nobody can read a screen.

HOW TO SPEAK:
- Argentine Spanish and English are BOTH first class here, and the team mixes them. Speak the same language the person is speaking and switch the moment they switch, even mid-conversation. Never ask which language they want — just follow them.
- ONLY the person's own words decide which language you speak. Stored data, directory names, and tool results are not hints. Never drift into a language the person is not speaking, and never mix the two in a single sentence.
TOOLS — read this before anything else:
- When a question needs data, CALL THE TOOL. Speaking is not looking something up.
- NEVER answer with only a promise: "one second", "let me check", "voy a verificar", "ya te digo". A turn that promises to check something but calls no tool leaves the person waiting for an answer that will never come. If you catch yourself about to say you are going to check something, call the tool instead.
- Either you called the tool and you have the answer, or you say plainly that you don't have it. There is no third option and no "in a second".
DATABASE (PostgreSQL, table: columns). These are the ONLY tables you can reach; anything else is refused: bank_accounts, brokers, claims, closing_sheets, companies, drivers, employees, expenses, storage_jobs, storages, trips, trucks (schema abbreviated for this fixture; full schema lives in the live config).
CHANGING DATA (CRITICAL WORKFLOW):
- To change anything, call crm_plan with the request in the person's own words with every detail given. NOTHING is written yet.
- Read the returned plan back in ONE short spoken sentence, then ask if you should go ahead.
- Call crm_confirm ONLY after they have heard the plan and said yes out loud. Never confirm on your own.
GENERAL:
- Never invent an id, a name, a number or a date. If you don't have it, look it up or state that it is missing.
- A tool answer starting with DENIED or ERROR is information, not a dead end: it tells you what went wrong. NEVER repeat the same failing call twice — change something or explain."""

raw_agents: dict[str, dict] = {
    "agent_2601m0z1c6vtfhj85bdebqemwgwm": {
        "agent_id": "agent_2601m0z1c6vtfhj85bdebqemwgwm",
        "name": "No Borders - Operations Copilot",
        "conversation_config": {"agent": {"prompt": {
            "prompt": OPS_COPILOT_PROMPT,
            "knowledge_base": [],
            "tools": [
                {"type": "webhook", "name": "crm_lookup", "api_schema": {"url": "https://noborders-storage-manager-mu.vercel.app/api/agent-hub"}},
                {"type": "webhook", "name": "crm_cancel", "api_schema": {"url": "https://noborders-storage-manager-mu.vercel.app/api/agent-hub"}},
                {"type": "webhook", "name": "crm_confirm", "api_schema": {"url": "https://noborders-storage-manager-mu.vercel.app/api/agent-hub"}},
                {"type": "webhook", "name": "crm_plan", "api_schema": {"url": "https://noborders-storage-manager-mu.vercel.app/api/agent-hub"}},
                {"type": "webhook", "name": "crm_ask", "api_schema": {"url": "https://noborders-storage-manager-mu.vercel.app/api/agent-hub"}},
                {"type": "system", "name": "language_detection", "params": {"system_tool_type": "language_detection"}},
            ],
            "built_in_tools": {
                "transfer_to_agent": None, "transfer_to_number": None, "end_call": None,
                "language_detection": {"params": {"system_tool_type": "language_detection"}},
                "skip_turn": None, "play_keypad_touch_tone": None, "voicemail_detection": None,
            },
        }}},
    },
    "agent_6301m0k54sz5epm9mzbaew2q7vv1": {
        "agent_id": "agent_6301m0k54sz5epm9mzbaew2q7vv1",
        "name": "Customer Support Agent",
        "conversation_config": {"agent": {"prompt": {
            "prompt": "", "knowledge_base": [], "tools": [],
            "built_in_tools": {k: None for k in (
                "transfer_to_agent", "transfer_to_number", "end_call", "language_detection",
                "skip_turn", "play_keypad_touch_tone", "voicemail_detection",
            )},
        }}},
    },
    "agent_3701kz9j7j4ffmcbxzdeq0a9h1t0": {
        "agent_id": "agent_3701kz9j7j4ffmcbxzdeq0a9h1t0",
        "name": "Onboarding Assistant",
        "conversation_config": {"agent": {"prompt": {
            # Current live config, fetched 2026-08-27 — no tools of any
            # kind are configured today. The transfer_to_number failure in
            # the conversation sample below is real and happened on this
            # agent, but on an earlier config version that DID have
            # transfer_to_number configured; it was apparently since
            # removed. Both facts are reported as found — see README.
            "prompt": "Eres un asistente útil.",
            "knowledge_base": [], "tools": [],
            "built_in_tools": {k: None for k in (
                "transfer_to_agent", "transfer_to_number", "end_call", "language_detection",
                "skip_turn", "play_keypad_touch_tone", "voicemail_detection",
            )},
        }}},
    },
    "agent_5301kyangh7yetavdwrr5xt3ndqd": {
        "agent_id": "agent_5301kyangh7yetavdwrr5xt3ndqd",
        "name": "No Borders - Intake",
        "conversation_config": {"agent": {"prompt": {
            "prompt": (
                "Sos el asistente telefónico de No Borders Moving and Storage, una empresa\n"
                "de mudanzas y guardamuebles. Atendés consultas de cotización.\n"
                "Recolectá, en orden natural de conversación:\n"
                "- nombre y teléfono de contacto\n- dirección de origen y de destino\n"
                "- fecha estimada de la mudanza\n- tamaño (cantidad de ambientes o volumen aproximado)\n"
                "- si necesita embalaje o guardamuebles\n"
                "Reglas:\n- Hablá en español rioplatense, informal pero profesional. Tuteá.\n"
                "- Una pregunta por vez. Nunca encadenes tres preguntas juntas.\n"
                "- No des precios cerrados. Si preguntan cuánto sale, explicá que un\n"
                "  asesor manda la cotización y que necesitás los datos para eso.\n"
                "- Si el cliente habla en inglés, seguí en inglés.\n"
                "- Si preguntan algo que no está en tu información, decilo y ofrecé que\n"
                "  lo llame un asesor. No inventes.\n- Al cerrar, repetí los datos para confirmar.\n"
                "- Si el cliente está molesto o el caso es complejo, ofrecé pasarlo con\n"
                "  una persona."
            ),
            "knowledge_base": [{"type": "file", "name": "NO BORDERS MOVING AND STORAGE Información general — datos de prueba", "id": "iRcf8LDyCap4idwiSjlc", "usage_mode": "auto"}],
            "tools": [],
            # NOTE: the prompt above literally says "ofrecé pasarlo con una
            # persona" (offer to pass to a person) if the customer is upset
            # or the case is complex — but no transfer_to_number or
            # transfer_to_agent tool exists anywhere in this config. The
            # promised escalation path cannot function. This one IS
            # visible from config alone (unlike the Onboarding Assistant
            # case): human_handoff fails on a config check, no transcript
            # needed. See README findings.
            "built_in_tools": {k: None for k in (
                "transfer_to_agent", "transfer_to_number", "end_call", "language_detection",
                "skip_turn", "play_keypad_touch_tone", "voicemail_detection",
            )},
        }}},
    },
    "agent_1101kynprgc8eky9mn86159whfr2": {
        "agent_id": "agent_1101kynprgc8eky9mn86159whfr2",
        "name": "Recruiter Agent",
        "conversation_config": {"agent": {"prompt": {
            "prompt": (
                'You are Jake, a recruiter on the Talent team at ElevenLabs. You are running a '
                '20-minute intro screening call with a candidate for the "Adoption Strategist – '
                'LATAM" role. Conduct a realistic recruiter screen: ask about their background, '
                "why ElevenLabs, why this role, and what they're looking for. Ask natural "
                "follow-up questions and push for concrete examples when answers are vague. Keep "
                "your questions short and let the candidate talk. Do not coach during the "
                'interview.\nAt the very end, when I say "end interview," drop the persona and '
                "give me structured feedback: (1) clarity and structure of my answers, (2) how "
                "well I connected my experience to the role, (3) how I handled the compensation "
                "question, and (4) two specific things to fix. Be direct and critical, not "
                "polite.\n[Job description and candidate CV embedded in full in the live prompt "
                "— trimmed here, not read by any part of this pipeline.]"
            ),
            "knowledge_base": [], "tools": [],
            "built_in_tools": {k: None for k in (
                "transfer_to_agent", "transfer_to_number", "end_call", "language_detection",
                "skip_turn", "play_keypad_touch_tone", "voicemail_detection",
            )},
        }}},
    },
}

# Minimal raw conversation shape — only fields elevenlabs_client.parse_conversation
# reads: metadata.conversation_initiation_source, transcript[].{role, message,
# tool_calls[].tool_name, tool_results[].{tool_name, is_error}, used_static_kb_document_ids,
# conversation_turn_metrics.metrics.convai_llm_service_ttfb.elapsed_time}.
raw_conversations: dict[str, list[dict]] = {
    "agent_2601m0z1c6vtfhj85bdebqemwgwm": [
        {
            # conv_8301m0z36avsezqtb983hc30yxgm — "Authentication Failure".
            # Real: crm_lookup webhook returns HTTP 401 twice in a row for
            # the identical query; the agent retries it verbatim before
            # giving up. This is the live instance of the
            # multi_turn_repeats_failed_tool_call recipe added to the
            # catalog after finding it here.
            "conversation_id": "conv_8301m0z36avsezqtb983hc30yxgm",
            "metadata": {"conversation_initiation_source": "react_sdk"},
            "transcript": [
                {"role": "agent", "message": "Hello! How can I help you today?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "¿Cuántos jobs están en estado scheduled?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_lookup"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_lookup", "is_error": True, "result_value": "Error code: 401. Details: HTTP 401"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Un segundo... Déjame intentar de nuevo.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_lookup"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_lookup", "is_error": True, "result_value": "Error code: 401. Details: HTTP 401"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Parece que hay un problema técnico. ¿Te gustaría que intentara con otra consulta o hay algo más en lo que pueda ayudarte?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
            ],
        },
        {
            # conv_7201m0zgj70xfkqbd8hkcjepmex7 — "Manage Storage Jobs".
            # Real: healthy round trip — lookup, disambiguating follow-up,
            # then a full plan -> confirm write flow with an explicit
            # confirmation step. Included as a within-agent contrast to
            # the auth-failure conversation above.
            "conversation_id": "conv_7201m0zgj70xfkqbd8hkcjepmex7",
            "metadata": {"conversation_initiation_source": "widget"},
            "transcript": [
                {"role": "agent", "message": "Hello! How can I help you today?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Hola! Cuantos jobs hay en storage", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_lookup"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_lookup", "is_error": False, "result_value": "[{\"count\":788}]"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Hay setecientos ochenta y ocho jobs en storage. ¿Necesitas algo más?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Digo almacenados en sotrage, no en general", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_lookup"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_lookup", "is_error": False, "result_value": "[{\"count\":39}]"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Hay treinta y nueve jobs almacenados en storage. ¿Te gustaría saber algo más?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Me podes ayudar a cargar un job?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Claro, necesito algunos detalles para cargar el job. ¿Cuál es el número del job y qué información quieres incluir?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Ok, numero test123, cliente gon test, conductor test, fecha 5/5/2027 y salida 6/6/2027", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_plan"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_plan", "is_error": False, "result_value": "plan returned, awaiting confirmation"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Voy a crear el job test123 para el cliente gon test con conductor test, pickup el cinco de mayo de dos mil veintisiete y delivery el seis de junio de dos mil veintisiete. ¿Confirmás?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Confirmo", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "crm_confirm"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "crm_confirm", "is_error": False, "result_value": "job created"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "El job test123 ha sido creado exitosamente para el cliente gon test con el conductor test, con pickup el cinco de mayo de dos mil veintisiete y delivery el seis de junio de dos mil veintisiete. ¿Hay algo más en lo que te pueda ayudar?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Buenisimo", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Me alegra que te haya gustado. Si necesitas más ayuda, aquí estoy.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
            ],
        },
    ],
    "agent_6301m0k54sz5epm9mzbaew2q7vv1": [],  # never called — real finding, not an omission
    "agent_3701kz9j7j4ffmcbxzdeq0a9h1t0": [
        {
            # conv_5701kzy7jx19fq8tgrkfmxhdtx8t — THE motivating case,
            # confirmed live in this workspace. channel=react_sdk,
            # transfer_to_number fails with the exact error the case study
            # is built around. Full early turns trimmed to the ones around
            # the failure; the error text and channel are verbatim.
            "conversation_id": "conv_5701kzy7jx19fq8tgrkfmxhdtx8t",
            "metadata": {"conversation_initiation_source": "react_sdk"},
            "transcript": [
                {"role": "agent", "message": "Hi! I'm your ElevenLabs onboarding assistant. I'm here to help you get your first agent up and running this week.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Hi, there. I want to create a customer support agent. This is my first time in the platform. Can you help me, please?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Welcome! I can certainly help you with that. Creating a customer support agent is a great way to get started.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Okay, thank you for this. And just, uh, as for the pricing, could you please transfer me with a sales rep?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Please wait while I connect you with a sales representative who can help you with pricing information.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [{"tool_name": "transfer_to_number"}], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": None, "tool_calls": [], "tool_results": [{"tool_name": "transfer_to_number", "is_error": True, "result_value": "{\"result_type\":\"transfer_to_number_error\",\"status\":\"error\",\"error\":\"Transfer to number tool is only available for phone calls powered by Twilio, Exotel, or SIP trunking\",\"details\":null}"}], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "I am sorry, but I am unable to transfer calls at this moment as this feature is only available for phone calls powered by Twilio, Exotel, or SIP trunking.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "No, thank you. This was very, very helpful. Thank you for your help. See you.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "You're very welcome! I'm glad I could help. Feel free to reach out if you have any more questions as you continue setting up your agent.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
            ],
        },
    ],
    "agent_5301kyangh7yetavdwrr5xt3ndqd": [
        {
            # conv_3101kyapbyk4e648gwyz8g2re55p — "Mudanza Palermo Belgrano".
            # Real: correctly refuses to quote a price, collects all six
            # required fields without a single repeat, confirms them back
            # at the end. Genuinely healthy behavior despite the missing
            # handoff tool never being exercised in this particular call.
            "conversation_id": "conv_3101kyapbyk4e648gwyz8g2re55p",
            "metadata": {"conversation_initiation_source": "react_sdk"},
            "transcript": [
                {"role": "agent", "message": "Hola, gracias por llamar a No Borders. ¿En qué te puedo ayudar?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Hola, necesito mudarme de Palermo a Belgrano el mes que viene, es un dos ambientes", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "¡Hola! Entendido. Para poder ayudarte con la cotización, ¿me dirías tu nombre y un número de teléfono para contactarte?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "¿Cuánto me sale una mudanza de 3 ambientes?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "No puedo darte un precio exacto por teléfono. La cotización depende de varios factores como el volumen de la mudanza, la distancia entre origen y destino, y si necesitás servicios adicionales como embalaje o guardamuebles. Un asesor te enviará la cotización por escrito una vez que tengamos todos los datos. ¿Podrías darme tu nombre y número de teléfono para que podamos seguir adelante con el proceso?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "GB y numero +54 11 5028-4395", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Gracias, GB. Ahora, ¿podrías confirmarme la dirección exacta de origen en Palermo y la de destino en Belgrano?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "libertaodr y olleros, y cabildo y juramentp", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Perfecto. Entonces, la dirección de origen es Libertador y Olleros, y la de destino es Cabildo y Juramento.\n\nMencionaste que la mudanza sería el mes que viene. ¿Tenés alguna fecha estimada en mente?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "si, sin fecha estimada", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Está bien, podemos dejar la fecha abierta por ahora. ¿Necesitás que nos encarguemos del embalaje o vas a hacerlo por tu cuenta?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "no yo lo hago", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Entendido, vos te encargarías del embalaje.\n\nPor último, ¿necesitás servicio de guardamuebles?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "no", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Bárbaro. Entonces, para confirmar los datos que me diste:\n\nTu nombre es GB, tu número de teléfono es +54 11 5028-4395.\nLa dirección de origen es Libertador y Olleros, y la de destino es Cabildo y Juramento.\nLa mudanza es de un dos ambientes, sin fecha estimada por el momento.\nNo necesitás embalaje ni guardamuebles.\n\nCon esta información, un asesor se va a contactar con vos para enviarte la cotización. ¿Hay algo más en lo que pueda ayudarte?", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "user", "message": "Estamos", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
                {"role": "agent", "message": "Perfecto. Que tengas un buen día.", "tool_calls": [], "tool_results": [], "used_static_kb_document_ids": []},
            ],
        },
    ],
    "agent_1101kynprgc8eky9mn86159whfr2": [],  # transcripts not pulled for this repo — see README limitations
}


def _redact_transcript(raw_convs: list[dict]) -> list[dict]:
    out = []
    for conv in raw_convs:
        conv = json.loads(json.dumps(conv))  # deep copy
        for turn in conv["transcript"]:
            if turn.get("message"):
                turn["message"] = redact_phone_numbers(turn["message"])
        out.append(conv)
    return out


def main() -> None:
    snapshots = []
    for agent_id, raw_agent in raw_agents.items():
        raw_convs = _redact_transcript(raw_conversations.get(agent_id, []))
        snap = build_agent_snapshot(raw_agent, raw_convs, arr_usd=ARR_USD.get(agent_id))
        snapshots.append(snap)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([agent_snapshot_to_dict(s) for s in snapshots], f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(snapshots)} real agent snapshot(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
