"""
Runtime DecisionEngine class that mirrors the class used in the notebook
so that unpickling a model saved from a notebook (module "__main__") can
be safely mapped to this module.

NOTE: Unpickling does not call __init__; attributes are restored from the
pickle. We keep a compatible signature and attributes to match the pickled
object shape.
"""

from typing import Dict, Any

try:
    # Optional imports for completeness. These are used by __call__ in the notebook class.
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore

import numpy as np  # type: ignore


class DecisionEngine:
    def __init__(self, nlp_model_path: str, stress_model_path: str):
        # Stored in the notebook to locate the NLP model and stress model pickle
        self.nlp_model_path = nlp_model_path
        self.stress_model_path = stress_model_path
        # The notebook sets this after loading via pickle; keep attribute here for shape compatibility
        self.stress_model = None
        # Feature order as per notebook (ML Component 3)
        self.feature_order = [
            "Sleep_Duration",
            "BMI_Category",
            "Heart_Rate",
            "Daily_Steps",
            "Systolic_BP",
        ]

    def recommend_break(self, features: Dict[str, Any]) -> str:
        if features.get("Daily_Steps", 0) < 4000:
            return "Take a 10-minute walk to refresh your mind."
        elif features.get("Sleep_Duration", 0) < 6:
            return "Power down with a short rest or breathing exercise."
        elif features.get("Heart_Rate", 0) > 90:
            return "Try 5 minutes of mindfulness or meditation to calm down."
        else:
            return "Take a short coffee/tea break."

    def __call__(self, email_text: str, stress_features: Dict[str, Any]):
        # Load NLP model and tokenizer as in the notebook definition
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "Transformers not available to run NLP inference inside DecisionEngine.__call__"
            )

        tokenizer = AutoTokenizer.from_pretrained(self.nlp_model_path)
        nlp_model = AutoModelForSequenceClassification.from_pretrained(self.nlp_model_path)

        tokens = tokenizer(email_text, padding=True, truncation=True, return_tensors="pt")
        outputs = nlp_model(**tokens)
        intent_id = int(outputs.logits.argmax(dim=1).item())
        intent_label = nlp_model.config.id2label[intent_id]

        # Stress level prediction from sklearn model stored in self.stress_model
        stress_input = np.array([[stress_features[f] for f in self.feature_order]])
        if self.stress_model is None:
            raise RuntimeError("stress_model is None on DecisionEngine; unpickled object may be incomplete")
        stress_level = int(self.stress_model.predict(stress_input)[0])

        break_suggestion = self.recommend_break(stress_features)

        if "high" in intent_label.lower() and stress_level >= 7:
            recommendation = f"You seem stressed. {break_suggestion} Then tackle the high-priority task."
        elif "high" in intent_label.lower():
            recommendation = "You’re in a good state. Start working on this high-priority task."
        elif "reminder" in intent_label.lower() and stress_level >= 7:
            recommendation = f"Handle the reminder later. {break_suggestion}"
        else:
            recommendation = f"Intent: {intent_label}, Stress Level: {stress_level}. {break_suggestion}"

        return {
            "intent": intent_label,
            "stress_level": stress_level,
            "recommendation": recommendation,
        }
