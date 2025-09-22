"""Custom Qwen Model w/ ToolUniverse added via Qwen-Agent"""
from __future__ import annotations
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from eval_framework import BaseModel  # type: ignore

logger = logging.getLogger(__name__)


TOOLUNIVERSE_MAX_STEPS = 6
DEFAULT_MAX_TOOLS = int(os.getenv("CURE_MAX_TOOLS", "25"))  # limit tools injected into schema to control context
DEFAULT_TEMPERATURE = float(os.getenv("CURE_TEMP", "0.2"))


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: Dict[str, Any]
    result_preview: str
    full_result: Any


class QwenAgentToolUniverseModel(BaseModel):
    """Model wrapper using Qwen-Agent function calling with ToolUniverse tools."""

    def __init__(self, model_name: str, device: str = "auto", max_tools: Optional[int] = None, temperature: Optional[float] = None):
        super().__init__(model_name)
        self.device = device
        self.llm = None  # Qwen-Agent model instance
        self.tool_engine = None
        self.functions: List[Dict[str, Any]] = []
        self.tool_usage: List[ToolCallRecord] = []
        self.max_tools = max_tools if (max_tools is not None and max_tools > 0) else DEFAULT_MAX_TOOLS
        self.temperature = temperature if (temperature is not None and temperature >= 0) else DEFAULT_TEMPERATURE

    def load(self, **kwargs):  # noqa: D401
        try:
            # Build Qwen-Agent transformers model
            from qwen_agent.llm import get_chat_model
            from tooluniverse.execute_function import ToolUniverse
            # Load tool universe
            self.tool_engine = ToolUniverse()
            self.tool_engine.load_tools()
            tools = self.tool_engine.return_all_loaded_tools()
            # Truncate tool list if too large (helps small models + token budget)
            if len(tools) > self.max_tools:
                logger.info(f"Truncating tools from {len(tools)} to max_tools={self.max_tools}")
                tools = tools[: self.max_tools]
            # Convert to OpenAI-style function schema (lightweight properties only)
            fn_list: List[Dict[str, Any]] = []
            for t in tools:
                raw_params = t.get('parameter') or {}
                # Sanitize: ensure each property has at least a 'type'
                properties = {}
                if isinstance(raw_params, dict):
                    for p_name, p_def in raw_params.items():
                        if isinstance(p_def, dict):
                            pd = dict(p_def)
                            if 'type' not in pd:
                                pd['type'] = 'string'
                            # Hard truncate long descriptions
                            if 'description' in pd and isinstance(pd['description'], str):
                                pd['description'] = pd['description'][:200]
                            properties[p_name] = pd
                        else:
                            properties[p_name] = {'type': 'string'}
                self_desc = t.get('description', '')
                fn_list.append({
                    'name': t['name'],
                    'description': self_desc[:400] if isinstance(self_desc, str) else '',
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                        'required': t.get('required', []) if isinstance(t.get('required', []), list) else []
                    }
                })
            self.functions = fn_list
            # Instantiate the local transformers model via Qwen-Agent
            self.llm = get_chat_model({
                'model': self.model_name,
                'model_type': 'transformers',
                'generate_cfg': {
                    'max_new_tokens': kwargs.get('max_new_tokens', 512),
                    'fncall_prompt_type': kwargs.get('fncall_prompt_type', 'nous'),
                    'parallel_function_calls': True,  # hint to prompt template
                    'function_choice': 'auto',
                    'top_p': 0.9,
                    'temperature': float(kwargs.get('temperature', self.temperature)),
                }
            })
            logger.info(f"Qwen-Agent transformers model + {len(self.functions)} tool schemas loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen-Agent ToolUniverse model: {e}") from e

    def _messages_to_dict(self, msgs):
        out = []
        for m in msgs:
            d = {}
            for k in ['role', 'content', 'function_call', 'reasoning_content']:
                v = getattr(m, k, None) if not isinstance(m, dict) else m.get(k)
                if v:
                    d[k] = v if isinstance(v, (str, dict)) else v
            out.append(d)
        return out

    def inference(self, prompt: str, max_tokens: int = 512) -> Tuple[str, List[Dict]]:  # noqa: D401
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        self.tool_usage.clear()
        # Start conversation
        messages: List[Dict[str, Any]] = [
            {'role': 'user', 'content': prompt}
        ]
        reasoning_trace: List[Dict[str, Any]] = [{
            'idx': 0,
            'role': 'user',
            'content': prompt,
            'type': 'user_input'
        }]
        final_answer: str = ''

        for step in range(1, TOOLUNIVERSE_MAX_STEPS + 1):
            try:
                responses = self.llm.chat(messages=messages, functions=self.functions, stream=False)
            except Exception as e:
                logger.error(f"ToolUniverseAgent llm.chat exception at step {step}: {e}")
                final_answer = f"(Model error at step {step}: {e})"
                break
            converted = self._messages_to_dict(responses)
            for c in converted:
                c['idx'] = len(reasoning_trace)
                if 'function_call' in c:
                    c['type'] = 'assistant_fncall'
                elif c.get('role') == 'assistant':
                    c['type'] = 'assistant_output'
                else:
                    c['type'] = 'other'
                reasoning_trace.append(c)
            messages.extend([r.model_dump() if hasattr(r, 'model_dump') else r for r in responses])

            fn_calls = []
            for r in responses:
                fc = getattr(r, 'function_call', None) if not isinstance(r, dict) else r.get('function_call')
                if fc:
                    fn_calls.append(fc)
            if not fn_calls:
                assistant_texts = []
                for r in responses:
                    fc = getattr(r, 'function_call', None) if not isinstance(r, dict) else r.get('function_call')
                    if not fc:
                        content = getattr(r, 'content', None) if not isinstance(r, dict) else r.get('content')
                        if content:
                            assistant_texts.append(content)
                final_answer = '\n'.join(assistant_texts).strip()
                break

            for fc in fn_calls:
                name = fc.get('name') if isinstance(fc, dict) else getattr(fc, 'name', '')
                try:
                    args_json = fc.get('arguments') if isinstance(fc, dict) else getattr(fc, 'arguments', '{}')
                    if isinstance(args_json, str):
                        try:
                            args = json.loads(args_json)
                        except Exception:
                            args = {}
                    else:
                        args = args_json
                except Exception:
                    args = {}
                exec_start = time.time()
                try:
                    # Preferred execution style: name + kwargs
                    if isinstance(args, dict):
                        tool_output = self.tool_engine.run_one_function(name, **args)
                    else:
                        tool_output = self.tool_engine.run_one_function(name)
                except Exception as te:
                    tool_output = f"Tool execution error: {te}"
                exec_dur = time.time() - exec_start
                preview = str(tool_output)
                if len(preview) > 500:
                    preview = preview[:500] + '...'
                self.tool_usage.append(ToolCallRecord(step=step, name=name, arguments=args if isinstance(args, dict) else {}, result_preview=preview, full_result=tool_output))
                func_msg = {
                    'role': 'function',
                    'name': name,
                    'content': preview,
                }
                messages.append(func_msg)
                reasoning_trace.append({
                    'idx': len(reasoning_trace),
                    'role': 'function',
                    'name': name,
                    'content': preview,
                    'type': 'tool_result',
                    'exec_time_s': round(exec_dur, 3)
                })
            if step == TOOLUNIVERSE_MAX_STEPS and not final_answer:
                final_answer = '(Max tool steps reached – partial answer withheld)'
        if not final_answer:
            final_answer = '(No direct answer produced)'
        reasoning_trace.append({
            'idx': len(reasoning_trace),
            'role': 'assistant',
            'content': final_answer,
            'type': 'final_answer'
        })
        return final_answer, reasoning_trace

    def export_tool_usage(self) -> List[Dict[str, Any]]:
        return [
            {
                'step': r.step,
                'name': r.name,
                'arguments': r.arguments,
                'result_preview': r.result_preview,
            } for r in self.tool_usage
        ]
