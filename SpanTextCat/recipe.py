import spacy
import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.components.loaders import JSONL
from prodigy.models.matcher import PatternMatcher
from prodigy.components.db import connect
from prodigy.util import set_hashes


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str),
    patterns=("The match patterns file", "option", "p", str),
    exclusive=("Treat classes as mutually exclusive", "flag", "E", bool),
)
def custom_recipe(dataset, lang, file_in, patterns,  exclusive=False,):
    span_labels = ["PRECONDITION", "POSTCONDITION",
                   "ACTOR", "QUALITY", "ACTION"]
    textcat_labels = ["NONE", "PRECONDITION", "POSTCONDITION", "BOTH"]

    def add_options(stream):
        for ex in stream:
            ex['options'] = [
                {"id": lab, "text": lab} for lab in textcat_labels
            ]
            yield ex

    def remove_duplication(stream):
        input_hashes = connect().get_input_hashes(dataset)
        for eg in stream:
            eg = set_hashes(eg)
            if eg["_input_hash"] not in input_hashes:
                yield eg

    nlp = spacy.blank(lang)
    stream = JSONL(file_in)

    if patterns is not None:
        pattern_matcher = PatternMatcher(
            nlp, combine_matches=True, all_examples=True)
        pattern_matcher = pattern_matcher.from_disk(patterns)
        stream = (eg for _, eg in pattern_matcher(stream))

    stream = add_tokens(nlp, stream, use_chars=None)

    stream = add_options(stream)
    stream = remove_duplication(stream)
    blocks = [
        {"view_id": "spans_manual"},
        {"view_id": "choice", "text": None},
    ]
    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples,
        'exclude': [dataset],
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": span_labels,
            "blocks": blocks,
            "keymap_by_label": {
                "0": "q",
                "1": "w",
                "2": "e",
                "3": "r",
                "PRECONDITION": "1",
                "POSTCONDITION": "2",
                "ACTOR": "3",
                "QUALITY": "4",
                "ACTION": "5"
            },
            "choice_style": "single" if exclusive else "multiple",  # Style of choice interface,
            # "exclude_by": "input"
        },
    }
