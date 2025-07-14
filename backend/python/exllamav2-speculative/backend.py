#!/usr/bin/env python3
import grpc
from concurrent import futures
import time
import backend_pb2
import backend_pb2_grpc
import argparse
import signal
import sys
import os
import glob

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import version as torch_version


from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler
)


from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init,
)


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))

# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self):
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.cache = None
        # Draft model support (following TabbyAPI pattern)
        self.draft_model = None
        self.draft_cache = None
        
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))

    def LoadModel(self, request, context):
        try:
            model_directory = request.ModelFile

            # Main model configuration
            config = ExLlamaV2Config()
            config.model_dir = model_directory
            if request.ContextSize > 0:
                config.max_seq_len = request.ContextSize
            config.prepare()

            model = ExLlamaV2(config)
            cache = ExLlamaV2Cache(model, lazy=True)
            model.load_autosplit(cache)

            # Draft model setup (following TabbyAPI pattern)
            draft_model = None
            draft_cache = None
            
            if request.DraftModel and request.DraftModel.strip():
                # Handle relative paths like TabbyAPI does
                if not os.path.isabs(request.DraftModel):
                    draft_model_path = os.path.join(os.path.dirname(model_directory), request.DraftModel)
                else:
                    draft_model_path = request.DraftModel
                    
                draft_config = ExLlamaV2Config()
                draft_config.model_dir = draft_model_path
                # Use same context size as main model (TabbyAPI approach)
                draft_config.max_seq_len = config.max_seq_len
                # Default rope settings for draft model
                draft_config.scale_pos_emb = 1.0  # draft_rope_scale default
                draft_config.prepare()
                
                draft_model = ExLlamaV2(draft_config)
                draft_cache = ExLlamaV2Cache(draft_model, lazy=True)
                draft_model.load_autosplit(draft_cache)

            tokenizer = ExLlamaV2Tokenizer(config)

            # Use dynamic generator when draft model is available, otherwise base generator
            if draft_model is not None:
                generator = ExLlamaV2DynamicGenerator(
                    model=model,
                    cache=cache,
                    tokenizer=tokenizer,
                    draft_model=draft_model,
                    draft_cache=draft_cache
                )
            else:
                # Fallback to base generator for backward compatibility
                generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

            self.generator = generator
            self.model = model
            self.draft_model = draft_model
            self.tokenizer = tokenizer
            self.cache = cache
            self.draft_cache = draft_cache

            generator.warmup()
            
        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        return backend_pb2.Result(message="Model loaded successfully", success=True)

    def Predict(self, request, context):

        penalty = 1.15
        if request.Penalty != 0.0:
            penalty = request.Penalty

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = request.Temperature
        settings.top_k = request.TopK
        settings.top_p = request.TopP
        settings.token_repetition_penalty = penalty
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        tokens = 512

        if request.Tokens != 0:
            tokens = request.Tokens
        output = self.generator.generate_simple(
            request.Prompt, settings, tokens)

        # Remove prompt from response if present
        if request.Prompt in output:
            output = output.replace(request.Prompt, "")

        return backend_pb2.Result(message=bytes(output, encoding='utf-8'))

    def PredictStream(self, request, context):
        # Implement PredictStream RPC
        # for reply in some_data_generator():
        #    yield reply
        # Not implemented yet
        return self.Predict(request, context)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ])
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    serve(args.addr)
