import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode

from flask import Flask, Response, request
from twilio.twiml.voice_response import Gather, VoiceResponse

from common import append_jsonl, ensure_dir, load_yaml

app = Flask(__name__)
_FLOW: List[Dict[str, str]] = []
_RESPONSES_PATH: Path
_EVENTS_PATH: Path
_VOICE = "alice"
_LANGUAGE = "en-IN"
_ESTIMATED_MINUTES = 8
_MAX_WORDS_PER_UTTERANCE = 10


def load_flow(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return []
    out: List[Dict[str, str]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        question_type = str(item.get("type", "open")).strip()
        if question_id and prompt:
            out.append({"id": question_id, "prompt": prompt, "type": question_type})
    return out


def twiml_response(response: VoiceResponse) -> Response:
    return Response(str(response), mimetype="application/xml")


def _record_event(payload: Dict[str, object]) -> None:
    append_jsonl(_EVENTS_PATH, [payload])


def _split_for_speech(text: str) -> List[str]:
    compact = " ".join(text.split())
    if not compact:
        return []
    sentence_parts = [part.strip() for part in re.split(r"[.!?;:]", compact) if part.strip()]
    out: List[str] = []
    for part in sentence_parts:
        words = part.split()
        while words:
            chunk = words[:_MAX_WORDS_PER_UTTERANCE]
            out.append(" ".join(chunk))
            words = words[_MAX_WORDS_PER_UTTERANCE:]
    return out


def _say_chunked(target: object, text: str) -> None:
    for chunk in _split_for_speech(text):
        target.say(chunk, voice=_VOICE, language=_LANGUAGE)


def _gather_for_step(step: int) -> Gather:
    return Gather(
        input="speech dtmf",
        num_digits=1,
        timeout=6,
        speech_timeout="auto",
        language=_LANGUAGE,
        action=f"/voice/step?{urlencode({'step': step})}",
        method="POST",
    )


@app.route("/voice/start", methods=["GET", "POST"])
def voice_start() -> Response:
    call_sid = request.values.get("CallSid", "")
    response = VoiceResponse()
    _say_chunked(
        response,
        f"Hi, and thank you for joining this research call. This interview takes around {_ESTIMATED_MINUTES} minutes. "
        "It is conversational, so you can reply naturally in your own words, or use keypad numbers when convenient. "
        "You will receive rupees two hundred after successfully completing the full survey. "
        "Compensation delivery method and timing will be shared after completion verification.",
    )
    gather = _gather_for_step(step=0)
    _say_chunked(
        gather,
        "Before we begin, do you consent to participate? You can say yes or no, or press 1 for yes and 2 for no.",
    )
    response.append(gather)
    _say_chunked(response, "I did not catch that response. We can try again later. Goodbye for now.")
    response.hangup()
    _record_event(
        {
            "event_type": "call_started",
            "call_sid": call_sid,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return twiml_response(response)


@app.route("/voice/step", methods=["POST"])
def voice_step() -> Response:
    call_sid = request.values.get("CallSid", "")
    digits = request.values.get("Digits", "")
    speech_result = request.values.get("SpeechResult", "")
    speech_confidence = request.values.get("Confidence", "")
    step = int(request.args.get("step", "0"))
    if step < len(_FLOW):
        question = _FLOW[step]
        response_value = digits or speech_result
        response_type = "dtmf" if digits else ("speech" if speech_result else "empty")
        append_jsonl(
            _RESPONSES_PATH,
            [
                {
                    "response_id": f"{call_sid}:{question['id']}",
                    "call_sid": call_sid,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "question_id": question["id"],
                    "question_type": question.get("type", "open"),
                    "response_value": response_value,
                    "response_digits": digits,
                    "response_speech": speech_result,
                    "speech_confidence": speech_confidence,
                    "response_type": response_type,
                    "survey_mode": "voice",
                }
            ],
        )

    next_step = step + 1
    response = VoiceResponse()
    if next_step >= len(_FLOW):
        _say_chunked(
            response,
            "Thank you. You have completed the survey. Your responses were saved successfully.",
        )
        response.hangup()
        _record_event(
            {
                "event_type": "call_completed",
                "call_sid": call_sid,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "steps_answered": next_step,
            }
        )
        return twiml_response(response)

    prompt = _FLOW[next_step]["prompt"]
    gather = _gather_for_step(step=next_step)
    _say_chunked(gather, prompt)
    response.append(gather)
    _say_chunked(response, "I could not capture an answer. We will end here and follow up by message if needed.")
    response.hangup()
    return twiml_response(response)


@app.route("/voice/status", methods=["POST"])
def voice_status() -> Response:
    _record_event(
        {
            "event_type": "call_status",
            "call_sid": request.values.get("CallSid", ""),
            "call_status": request.values.get("CallStatus", ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return Response("", status=204)


def run(config_path: str, host: str, port: int) -> None:
    global _FLOW, _RESPONSES_PATH, _EVENTS_PATH, _VOICE, _LANGUAGE, _ESTIMATED_MINUTES, _MAX_WORDS_PER_UTTERANCE

    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    flow_path = root / "config" / "twilio_question_flow.json"
    _FLOW = load_flow(flow_path)
    twilio_cfg = config.get("twilio", {})
    if isinstance(twilio_cfg, dict):
        _VOICE = str(twilio_cfg.get("voice_name", "alice"))
        _LANGUAGE = str(twilio_cfg.get("language", "en-IN"))
        _MAX_WORDS_PER_UTTERANCE = int(twilio_cfg.get("max_words_per_utterance", 10))
    survey_cfg = config.get("survey", {})
    if isinstance(survey_cfg, dict):
        _ESTIMATED_MINUTES = int(survey_cfg.get("estimated_duration_minutes", 8))

    path_cfg = config.get("paths", {})
    if not isinstance(path_cfg, dict):
        raise ValueError("Invalid paths config")
    _RESPONSES_PATH = root / str(path_cfg.get("responses_jsonl", "data/primary/survey_responses.jsonl"))
    _EVENTS_PATH = root / str(path_cfg.get("survey_events_jsonl", "data/primary/survey_events.jsonl"))
    ensure_dir(_RESPONSES_PATH.parent)
    ensure_dir(_EVENTS_PATH.parent)
    app.run(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Twilio webhook app for phone survey")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    run(config_path=args.config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
