import json
from datasets import load_dataset

template_mmlu = 'Question: {}\n(A) {} (B) {} (C) {} (D) {}\n Answer: \n Let\'s think step by step. '
template_gsm8k = 'Reason the math question below step by step. Question: {}.\n Answer: '
# cais/mmlu uses 'all' as the config name to load all subjects at once
MMLU_CONFIG = 'all'


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

def get_test_set(dataset_name):
    if dataset_name == 'mmlu':
        # cais/mmlu provides all subjects under config 'all', split 'test'
        ds = load_dataset('cais/mmlu', MMLU_CONFIG, split='test')
        return list(ds)
    elif dataset_name == 'sampled_mmlu':
        res = []
        with open(sample_file) as f:
            res = json.load(f)
        return res
    elif dataset_name == 'gsm8k':
        gsm8k = load_dataset('openai/gsm8k', 'main')
        return list(gsm8k['test'])

