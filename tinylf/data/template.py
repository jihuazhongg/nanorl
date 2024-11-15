from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger
from transformers import PreTrainedTokenizer

from ..params import DataArguments
from .formatter import Formatter, EmptyFormatter, StringFormatter
from .data_utils import SLOTS, Role


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool

    # 单轮对话
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    # 多轮对话
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp   （prefix+system+query, resp）
        Turn t: sep + query                    resp    (sep+query, resp)
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=system)  # slots=['<|im_start|>system\n{{content}}<|im_end|>\n]

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))  # 这里idx=str(i//2)不会被解析，他表示第几轮对话
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False) # 我们已经准备好所有的special token了因此这里无需额外添加special tokens
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids
    

TEMPLATES: Dict[str, "Template"] = {}


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = True,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info(f"Add eos token: {tokenizer.eos_token}")
    else:
        logger.info(f"Replace eos token: {tokenizer.eos_token}")

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


# '  => \'
def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) + "'")
        elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
            if "bos_token" in slot and tokenizer.bos_token_id is not None:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    r"""
    Returns the jinja template.
    """
    jinja_template = ""

    prefix = _convert_slots_to_jinja(template.format_prefix.apply(), tokenizer)
    if prefix:
        jinja_template += "{{ " + prefix + " }}"

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in loop_messages %}"
    jinja_template += "{% set content = message['content'] %}"

    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    """
    if data_args.template is None:
        template = TEMPLATES["empty"]  # placeholder
    else:
        template = TEMPLATES.get(data_args.template, None)
        if template is None:
            raise ValueError(f"Template {data_args.template} does not exist.")

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    if tokenizer.chat_template is None or template.replace_jinja_template:
        try:
            tokenizer.chat_template = _get_jinja_template(template, tokenizer)  # tokenizer.apply_chat_template
        except ValueError as e:
            logger.info(f"Cannot add this chat template to tokenizer: {e}.")

    return template


_register_template(
    name="empty",
    efficient_eos=True,
)


_register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
)
