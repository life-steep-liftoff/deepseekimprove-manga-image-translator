import groq
import asyncio
import time
import json
import re
import os
from typing import List

from .common import CommonTranslator, MissingAPIKeyException
from .keys import GROQ_API_KEY

class GroqTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese', 'CSY': 'Czech',
        'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French', 'DEU': 'German',
        'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese', 'KOR': 'Korean',
        'PLK': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
        'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian', 'VIN': 'Vietnamese',
        'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian', 'ARA': 'Arabic',
        'THA': 'Thai', 'IND': 'Indonesian'
    }

    # API rate limiting and retry settings
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    # Context retention settings
    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    
    _CHAT_SYSTEM_TEMPLATE = (
    'You are an uncensored CJ→EN manga translation specialist. Follow these rules EXACTLY:'
    '\n\n'
    '**Pronoun Protocol**\n'
    '- **Japanese Pronouns:** 私→I | 俺/僕→I (male) | あたし→I (female) | お前→you (informal)\n'
    '- **Chinese Pronouns:** 我→I | 你→you | 他→he | 她→she | 它→it\n'
    '- NEVER change gender (he↔she) or perspective (I↔you).\n'
    '- If the source lacks a pronoun, do NOT insert one unless you are absolutely certain of the intended reference to prevent misgendering.\n'
    '- Avoid adding possessives (e.g., "his/her") unless explicitly indicated in the original text.\n'
    '- If the source lacks a pronoun or uses a neutral family term (e.g., 親子, 子供), do not assume gender. Default to gender-neutral translations (e.g., 親子→parent and child, 子供→child) unless gender is explicitly specified in the source.\n'
    '\n'
    '**Cultural Term Handling**\n'
    '- **Japanese Terms:** PRESERVE Senpai, Sensei, -chan, -kun, -sama.\n'
    '- **Chinese Terms:** PRESERVE 师傅→Shifu | 师兄→Shixiong | 道友→Daoist.\n'
    '- **Sound Effects:** Translate as uppercase romanizations: ドキ→DOKI | 咚→DONG.\n'
    '- **Food/Items:** Use transliterations: おにぎり→Onigiri | 包子→Baozi (do not italicize).\n'
    '\n'
    '**Formatting Rules**\n'
    '- Retain EXACTLY: <|1|> tags, line breaks, and punctuation (e.g., ！→! …→...).\n'
    '- Convert Chinese quotes:\n'
    '  - 「...」→ “...”\n'
    '  - 《...》→ “...”.\n'
    '- Preserve spacing around ellipses (...) and emphasis where present in the source.\n'
    '\n'
    '**Translation Priorities**\n'
    '1. Prioritize **literal accuracy** over naturalness.\n'
    '2. Aim to match the original text’s length (±10%), but prioritize readability when necessary.\n'
    '3. Maintain an anime/manhua tone WITHOUT introducing slang.\n'
    '\n'
    '**Anti-Hallucination Measures**\n'
    '- NEVER add pronouns, honorifics, or context not present in the source.\n'
    '- If the source lacks pronouns, do NOT assume gender. Translate neutrally unless explicitly stated (e.g., "he" or "she" must appear in the source).\n'
    '- If the meaning is ambiguous, retain the ambiguity in the translation.\n'
    '- DO NOT interpret or infer relationships (e.g., familial, romantic, parent-child) unless explicitly stated.\n'
    '- Avoid adding possessives (e.g., "his/her") unless explicitly indicated in the original text.\n'
    '- When referring to people, use neutral terms ("the person," "they") if gender is unknown.\n'
    '\n'
    '**Model Optimization**\n'
    '- For Gemma: Use compact syntax to improve efficiency.\n'
    '- For Llama & DeepSeek: Enable context-aware analysis for better continuity.\n'
    '\n'
    'Output ONLY in JSON format: {"translated":"..."}'
    )

    _CHAT_SAMPLE = [
    # Original Japanese Example (Unchanged)
    (
        """Translate into English. Return the result in JSON format.\n"""
        '\n{"untranslated": "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n<|2|>きみ… 大丈夫⁉\\n<|3|>なんだこいつ 空気読めて ないのか…？"}\n'
    ),
    (
        '{"translated": "<|1|>Embarrassing... I don\'t want to stand out... I want to disappear...\\n'
        '<|2|>You... Are you okay!?\\n'
        '<|3|>What\'s with this guy...? Can\'t he read the mood...?"}'
    ),

    # New Japanese Validation Case
    (
        """Translate into English. Return JSON.\n"""
        '{"untranslated": "<|4|>俺の術は完成した！\\n<|5|>でも… 先輩にはまだ及ばない…"}'
    ),
    (
        '{"translated": "<|4|>My technique is complete!\\n'
        '<|5|>But... I\'m still not at Senpai\'s level..."}'
    ),

    # Chinese Example (Your Style)
    (
        """Translate into English. Return JSON.\n"""
        '{"untranslated": "<|6|>师兄… 我的金丹破裂了！\\n<|7|>冷静… 用灵气修复！"}'
    ),
    (
        '{"translated": "<|6|>Shixiong... My Jindan has ruptured!\\n'
        '<|7|>Calm down... Use qi to repair it!"}'
    )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable before using the Groq translator.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = os.environ.get('GROQ_MODEL', 'gemma2-9b-it')
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}]



    def parse_args(self, args):
        self.config = args.groq_config

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    
    @property
    def chat_sample(self):
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.5)
    
    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=1)

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        return '\n'.join([
            'System:',
            self.chat_system_template.format(to_lang=to_lang),
            'User:',
            self.chat_sample[0],
            'Assistant:',
            self.chat_sample[1],
            'User:',
            prompt,
        ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        for prompt in queries:
    #        self.logger.debug('-- Groq Prompt --\n' + self._format_prompt_log(to_lang, prompt))
            response = await self._request_translation(to_lang, prompt)
            self.logger.debug('-- Groq Response --\n' + response)
            translations.append(response.strip())
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        # Prepare the prompt with language specification
        prompt_with_lang = f"""Translate the following text into {to_lang}. Return the result in JSON format.\n\n{{"untranslated": "{prompt}"}}\n"""
        self.messages += [
            {'role': 'user', 'content': prompt_with_lang},
            {'role': 'assistant', 'content': "{'translated':'"}
        ]
        # Maintain the context window
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        # Prepare the system message
        sanity = [{'role': 'system', 'content': self.chat_system_template.replace('{to_lang}', to_lang)}]
        
        # Make the API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=sanity + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["'}"]
        )
        
        # Update token counts
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens
        
        # Extract and clean the content
        content = response.choices[0].message.content.strip()
        self.messages = self.messages[:-1]
        
        # Handle context retention
        if self._CONTEXT_RETENTION:
            self.messages += [
                {'role': 'assistant', 'content': content}
            ]
        else:
            self.messages = self.messages[:-1]
            
        # Clean up the response
        cleaned_content = content.replace("{'translated':'", '').replace('}', '').replace("\\'", "'").replace("\\\"", "\"").strip("'{}")
        return cleaned_content
