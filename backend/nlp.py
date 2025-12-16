from flask import Blueprint, request, jsonify
import traceback

try:
    from src.nlp.biobert import ClinicalNLP
except:
    from backend.src.nlp.biobert import ClinicalNLP

nlp_bp = Blueprint("nlp", __name__)

print("Loading NLP models, this may take a bit...")
nlp_model = None
_init_err = None
try:
    nlp_model = ClinicalNLP()
    print("NLP loaded")
except Exception as e:
    _init_err = str(e)
    print("Failed to load NLP:", _init_err)


@nlp_bp.route("/health", methods=["GET"])
def health():
    ok = nlp_model is not None
    return jsonify({"ok": ok, "loaded": ok, "error": None if ok else _init_err})


@nlp_bp.route("", methods=["POST"])
def analyze():
    if not nlp_model:
        return jsonify({"error": "NLP not ready", "detail": _init_err}), 500

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    question = payload.get("question", "")

    if not text:
        return jsonify({"error": "no text"}), 400

    ents = None
    try:
        ents = nlp_model.extract_entities(text)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "entity extraction failed", "detail": str(e)}), 500

    ans = None
    if question:
        try:
            ans = nlp_model.answer_question(text, question)
        except Exception as e:
            ans = {"error": "QA failed", "detail": str(e)}

    return jsonify({"entities": ents, "answer": ans})
