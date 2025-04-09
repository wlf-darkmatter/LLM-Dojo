from torch.utils.data import Dataset
import json
import torch
import rich
from rich import console


class RlhfDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = file.readlines()
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        prompt = data["prompt"]
        chosen = data["chosen"]
        rejected = data["rejected"]

        chosen_full_text = f"{prompt}\n\n### Response:\n{chosen}"
        rejected_full_text = f"{prompt}\n\n### Response:\n{rejected}"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_full_tokens = self.tokenizer.encode(chosen_full_text, add_special_tokens=False)
        rejected_full_tokens = self.tokenizer.encode(rejected_full_text, add_special_tokens=False)

        _input = {
            "prompt": prompt_tokens,
            "chosen": chosen_full_tokens,
            "rejected": rejected_full_tokens,
        }
        return _input

    def __len__(self):
        return len(self.data_list)


def translate(tokenizer, list_tokens: list, list_mask: list = None, verbose=False):
    if isinstance(list_tokens, torch.Tensor):
        if list_mask is not None:
            assert list_tokens.ndim == list_mask.ndim
        if list_tokens.ndim == 1:
            list_tokens = [list_tokens]
            if list_mask is not None:
                list_mask = [list_mask]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    list_echo = []
    str_rule = "Translate"
    if list_mask is not None:
        iterator_mask = iter(list_mask)  # * 遮挡 mask
        str_rule = "Translate with Mask"
    for tokens in list_tokens:
        if list_mask is not None:
            mask = next(iterator_mask)
            tokens = tokens[mask]
        idxes = torch.where(tokens == pad_id)[0]
        if len(idxes) == 0:
            ret = tokenizer.decode(tokens)
        else:
            idx = idxes[0]
            ret = tokenizer.decode(tokens[: idx + 1])

        list_echo.append(ret)
    console = rich.console.Console()
    console.rule(str_rule)
    if verbose:
        for i, echo_i in enumerate(list_echo):
            console.rule(f"{i}")
            rich.print(echo_i)

    return list_echo
