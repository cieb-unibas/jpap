from transformers import pipeline

def detect_language(text: list[str], model_id_or_path: str = "juliensimon/xlm-v-base-language-id", load_from_disk: bool = False, min_score = 0.75):
    if isinstance(text, str):
        text = [text]
    if load_from_disk:
        """
        Implement how to load from disk
        """
    else:
        p = pipeline("text-classification", model = model_id_or_path)
        outputs = [i["label"] if i["score"] > min_score else None for i in p(text)]
    return outputs