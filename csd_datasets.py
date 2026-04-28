import errno
import json
import os
import time

load_dataset = None
concatenate_datasets = None

template_mmlu = 'Question: {}\n(A) {} (B) {} (C) {} (D) {}\n Answer: \n Let\'s think step by step. '
template_gsm8k = 'Reason the math question below step by step. Question: {}.\n Answer: '
# cais/mmlu uses 'all' as the config name to load all subjects at once
MMLU_CONFIG = 'all'
MMLU_SUBJECTS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions',
]


template_mmlu_vicuna = 'Answer the question step by step: Question: {}\n(A) {} (B) {} (C) {} (D) {}\n Answer: \n'
template_gsm8k_vicuna = 'Reason the math question step by step. Question: {}\n Answer: '



def format_initial_input(item, dataset_name):
    if dataset_name == 'gsm8k':
        initial_input = template_gsm8k.format(item['question'])
    elif dataset_name == 'mmlu' or 'mmlu' in dataset_name:
        # cais/mmlu: keys are 'question', 'choices' (list of 4), 'answer' (int)
        q, ch = item['question'], item['choices']
        initial_input = template_mmlu.format(q, ch[0], ch[1], ch[2], ch[3])
    return initial_input


def format_vicuna_input(item, dataset_name):
    try:
        from fastchat.model.model_adapter import get_conversation_template
    except ImportError:
        raise ImportError("fastchat is required for vicuna formatting: pip install fschat")
    if dataset_name == 'gsm8k':
        initial_input = template_gsm8k_vicuna.format(item['question'])
    elif dataset_name == 'mmlu' or 'mmlu' in dataset_name:
        initial_input = template_mmlu_vicuna.format(*[item[f] for f in mmlu_features[:-1]])
    conv = get_conversation_template("vicuna")
    conv.messages = []
    conv.append_message(conv.roles[0], initial_input)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    return prompt



sample_file = '/home/your_dir/sampled_mmlu.json'


def _load_dataset_with_retry(*args, **kwargs):
    global load_dataset
    if load_dataset is None:
        from datasets import load_dataset as hf_load_dataset

        load_dataset = hf_load_dataset

    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(*args, **kwargs)
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == attempts:
                raise
            sleep_s = min(2 ** (attempt - 1), 16)
            dataset_name = args[0] if args else "<unknown>"
            dataset_config = args[1] if len(args) > 1 else None
            cache_root = os.environ.get("HF_DATASETS_CACHE", "<unset>")
            print(
                f"Retrying load_dataset({dataset_name}, {dataset_config}) after "
                f"ESTALE in {cache_root} (attempt {attempt}/{attempts}, "
                f"sleep {sleep_s}s)"
            )
            time.sleep(sleep_s)

def get_test_set(dataset_name):
    if dataset_name == 'mmlu':
        # On clusters running HF offline mode, the synthetic 'all' config may
        # be absent even when every per-subject cache is present.
        try:
            ds = _load_dataset_with_retry('cais/mmlu', MMLU_CONFIG, split='test')
        except ValueError as exc:
            if "Couldn't find cache for cais/mmlu for config 'all'" not in str(exc):
                raise
            subject_sets = []
            for subject in MMLU_SUBJECTS:
                subject_ds = _load_dataset_with_retry('cais/mmlu', subject, split='test')
                if 'subject' not in subject_ds.column_names:
                    subject_ds = subject_ds.add_column('subject', [subject] * len(subject_ds))
                subject_sets.append(subject_ds)
            global concatenate_datasets
            if concatenate_datasets is None:
                from datasets import concatenate_datasets as hf_concatenate_datasets

                concatenate_datasets = hf_concatenate_datasets

            ds = concatenate_datasets(subject_sets)
        return list(ds)
    elif dataset_name == 'sampled_mmlu':
        res = []
        with open(sample_file) as f:
            res = json.load(f)
        return res
    elif dataset_name == 'gsm8k':
        gsm8k = _load_dataset_with_retry('openai/gsm8k', 'main')
        return list(gsm8k['test'])
