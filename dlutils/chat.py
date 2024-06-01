from typing import List
from dataclasses import dataclass, field
from copy import deepcopy
import re

from .lazy_tokenizer import LazyTokenizer


@dataclass
class ChatPart:
    content: str
    truncatable: bool = False
    truncation_side: str = 'right'
    min_tokens: int = 0
    # parts with higher priority will be first truncated
    truncate_priority: int = 0
    # below are fields used by factory; do not specify them yourself
    ids: List[int] | None = None
    # uuid: UUID = field(default_factory=uuid4)

    def __repr__(self):
        return '<ChatPart' + (' truncable' if self.truncatable else '')  + '>'
    
    def get_ids(self):
        return self.ids
    
    def try_to_truncate(self, to_truncate: int) -> int:
        to_truncate = min(to_truncate, len(self.ids) - self.min_tokens)
        if to_truncate <= 0:
            return 0
        if self.truncation_side == 'left':
            self.ids = self.ids[-len(to_truncate):]
        else:
            self.ids = self.ids[:len(to_truncate)]
        return to_truncate


@dataclass
class Chat:
    role: str = 'user'
    parts: List[ChatPart] = field(default_factory=list)

    def __repr__(self):
        return f'<Chat: {self.role}, {len(self.parts)} parts>'


@dataclass
class ChatInput:
    chats: List[Chat] = field(default_factory=list)
    # used by factory
    ids: List[int] = field(default_factory=list)

    def __repr__(self):
        return f'<ChatInput: {len(self.chats)} chats>'

    def prepare_ids(self):
        fake_chat = []
        mapping = dict()
        for i, chat in enumerate(self.chats):
            fake_content = ''
            for j, part in enumerate(chat.parts):
                mapping[(i, j)] = part
                fake_content += f'∇{i},{j}∇'
            fake_chat.append({'role': chat.role, 'content': fake_content})


class ChatFactory(LazyTokenizer):
    def __init__(self, pretrained: str, max_tokens: int):
        self.pretrained = pretrained
        self.max_tokens = max_tokens
        self.tokenizer_kwargs = {'legacy': False, 'use_fast': True}
    
    def process(self, chat_input: ChatInput, tokenize: bool):
        chat_input = deepcopy(chat_input)
        trun_parts = []
        for chat in chat_input.chats:
            for part in chat.parts:
                if part.truncatable:
                    trun_parts.append(part)
        trun_parts.sort(key=lambda p: -p.truncate_priority)

        for chat in chat_input.chats:
            for part in chat.parts:
                part.ids = self.tok(part.content)

        fake_chat = []
        mapping = dict()
        for i, chat in enumerate(chat_input.chats):
            fake_content = ''
            for j, part in enumerate(chat.parts):
                mapping[(i, j)] = part
                fake_content += f'∇{i},{j}∇'
            fake_chat.append({'role': chat.role, 'content': fake_content})
        template = self.remove_bos_eos(self.tokenizer.apply_chat_template(fake_chat, tokenize=False))

        pieces = []
        re_pat = re.compile(r'(∇(\d+),(\d+)∇)')
        while template:
            matches = re_pat.findall(template)
            if matches:
                first = matches[0]
                i, j = map(int, first[1:])
                left = template.index(first[0])
                if left != 0:
                    pieces.append(self.tok(template[:left]))
                template = template[left + len(first[0]):]
                pieces.append(mapping[(i, j)].get_ids)
            else:
                pieces.append(self.tok(template))
                break
        
        def construct_input_ids():
            return sum([piece if isinstance(piece, list) else piece() for piece in pieces], start=[])

        remove_left = len(construct_input_ids()) - (self.max_tokens - 2)
        truncate_idx = 0
        while remove_left > 2:
            if truncate_idx >= len(trun_parts):
                raise NotImplementedError
            remove_left -= trun_parts[truncate_idx].try_to_truncate(remove_left)
        
        final_ids = construct_input_ids()
        return self.add_bos_eos_ids(final_ids)
