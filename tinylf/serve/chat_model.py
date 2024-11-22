import asyncio
import gc
from threading import Thread
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

import torch.cuda
from transformers import is_torch_npu_available

from ..params import get_infer_args
from .base_engine import Response
from .hf_engine import HuggingfaceEngine


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""
    General class for chat models. Backed by huggingface or vllm engines.

    Supports both sync and async methods.
    Sync methods: chat(), stream_chat() and get_scores().
    Async methods: achat(), astream_chat() and aget_scores().
    """

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        self.engine_type = model_args.infer_backend
        if model_args.infer_backend == "huggingface":
            self.engine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        r"""
        Gets a list of responses of the chat model.
        """
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        r"""
        Asynchronously gets a list of responses of the chat model.
        """
        return await self.engine.chat(messages, system, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        r"""
        Gets the response token-by-token of the chat model.
        """
        generator = self.astream_chat(messages, system, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""
        Asynchronously gets the response token-by-token of the chat model.
        """
        async for new_token in self.engine.stream_chat(messages, system, **input_kwargs):
            yield new_token


def run_chat() -> None:
    try:
        import readline  # noqa: F401
    except ImportError:
        print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            gc.collect()
            if is_torch_npu_available():
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
