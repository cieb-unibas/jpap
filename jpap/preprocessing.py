def _define_targets(inputs):
    targets = sorted(list(set(inputs)))
    n_classes = len(targets)
    target_idx = {targets[i]: i for i in range(n_classes)}
    return target_idx

def encode_labels(inputs, return_label_dict = True):
    label_dict = _define_targets(inputs=inputs)
    encoded_labels = [label_dict[t] for t in inputs]
    if return_label_dict:
        return encoded_labels, label_dict
    else:
        return encoded_labels

class PostingsTranslator():
    def __init__(self, target_language = "en"):
        self.target_language = target_language

    def load_translator():
        "Load a torch model to translate"

    def translate(self):
        "translate the text to english"
