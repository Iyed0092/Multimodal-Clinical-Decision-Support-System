import torch
from transformers import pipeline

NER_MODEL_NAME = "d4data/biomedical-ner-all"
QA_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1-squad"


class ClinicalNLP:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"Loading clinical NLP models on {device_name}")

        try:
            self.ner_pipeline = pipeline(
                task="ner",
                model=NER_MODEL_NAME,
                tokenizer=NER_MODEL_NAME,
                aggregation_strategy="simple",
                device=self.device
            )
            print("NER model ready")
        except Exception as e:
            print(f"NER initialization failed: {e}")
            self.ner_pipeline = None

        try:
            self.qa_pipeline = pipeline(
                task="question-answering",
                model=QA_MODEL_NAME,
                tokenizer=QA_MODEL_NAME,
                device=self.device
            )
            print("Q&A model ready")
        except Exception as e:
            print(f"Q&A initialization failed: {e}")
            self.qa_pipeline = None

    def extract_entities(self, text):
        if not text or self.ner_pipeline is None:
            return []

        results = self.ner_pipeline(text)
        entities = []

        for r in results:
            if r.get("score", 0) > 0.6:
                entities.append({
                    "text": r.get("word"),
                    "label": r.get("entity_group"),
                    "confidence": float(r.get("score"))
                })

        return entities

    def answer_question(self, context, question):
        if not context or not question or self.qa_pipeline is None:
            return "No context provided."

        result = self.qa_pipeline(
            question=question,
            context=context
        )

        return {
            "answer": result.get("answer"),
            "score": result.get("score")
        }


if __name__ == "__main__":
    nlp = ClinicalNLP()

    note = (
        "Patient presents with severe chest pain and shortness of breath. "
        "History of hypertension. Prescribed Aspirin 100mg daily. "
        "CT scan shows no abnormalities."
    )

    print("\nNER test")
    for ent in nlp.extract_entities(note):
        print(f"{ent['text']} -> {ent['label']}")

    print("\nQ&A test")
    answer = nlp.answer_question(note, "What medication was prescribed?")
    print(f"Answer: {answer['answer']} (score={answer['score']:.2f})")
