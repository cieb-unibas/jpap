import pandas as pd

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


def subsample_df(
        df: pd.DataFrame, group_col: str, seed = 4082023,
        max_n_per_group : int = 500, max_n_overall: int = None
        ):
    group_counts = {
        k: n if n < max_n_per_group else max_n_per_group for k, n in 
        dict(df.groupby([group_col])[group_col].count()).items()
        }
    df_out = pd.DataFrame()
    for g, n in group_counts.items():
        tmp = df.loc[df[group_col] == g, :].sample(n = n, random_state = seed).reset_index(drop=True)
        df_out = pd.concat([df_out, tmp], axis = 0)
    if max_n_overall:
        if len(df_out) > max_n_overall:
            df_out = df_out.reset_index(drop=True).sample(max_n_overall)

    return df_out

class PostingsTranslator():
    def __init__(self, target_language = "en"):
        self.target_language = target_language

    def load_translator():
        "Load a torch model to translate"

    def translate(self):
        "translate the text to english"
