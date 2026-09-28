# A book's loss by context length in llama.cpp, beside hf_long_context_loss.py: Gemma 4 12B for G2 and Llama 3.1 8B
# for L3 (ModelFamilyParity.md 8.2 and 8.4).
#
# llama-perplexity cannot run this protocol: it tokenizes without parsing control tokens, scores only the second half
# of each window, and overwrites each window's first token with <bos>. So this drives llama.dll through its C API
# (ctypes, struct layouts from the llama.h of the same release): the same turn and joined book text as the other
# harnesses, tokenized with control tokens parsed, fed in 1024-token chunks through llama.cpp's KV cache, every
# position's logits read back, and the loss reported per band -- 8K bands, or the bands ending at --prefixes.
#
#   python llama_cpp_long_context_loss.py --llama C:/Users/ToddT/llama.cpp/b11216 --model <file.gguf>
#       --book Data/Datasets/PG19/raw/test/30312.txt --tokens 32768
#   python llama_cpp_long_context_loss.py --family llama --llama C:/Users/ToddT/llama.cpp/b11216 --model <file.gguf>
#       --book Data/Datasets/PG19/raw/test/30312.txt --tokens 69632 --prefixes 8192 16384 32768 65536 69632

import argparse
import ctypes
import os
import struct
import sys
from pathlib import Path

import numpy as np

from hf_long_context_loss import BOOK_TURN as GEMMA_BOOK_TURN, BAND, CHUNK, join_wraps

sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 2 ] / 'Llama' ) )
from hf_llama_long_context_loss import BOOK_TURN as LLAMA_BOOK_TURN


class Batch( ctypes.Structure ):
    _fields_ = [ ( 'n_tokens', ctypes.c_int32 ),
                 ( 'token', ctypes.POINTER( ctypes.c_int32 ) ),
                 ( 'embd', ctypes.POINTER( ctypes.c_float ) ),
                 ( 'pos', ctypes.POINTER( ctypes.c_int32 ) ),
                 ( 'n_seq_id', ctypes.POINTER( ctypes.c_int32 ) ),
                 ( 'seq_id', ctypes.POINTER( ctypes.POINTER( ctypes.c_int32 ) ) ),
                 ( 'logits', ctypes.POINTER( ctypes.c_int8 ) ) ]


# The parameter structs are only read and written at a few fields. On Windows x64 a struct wider than eight bytes
# is returned through a hidden pointer and passed by pointer to a copy, so a buffer at least as large as the real
# struct is ABI-compatible. Offsets are from llama.h at b11216.
class Parameters( ctypes.Structure ):
    _fields_ = [ ( 'bytes', ctypes.c_uint8 * 1024 ) ]

    def set_int32( self, offset: int, value: int ):
        struct.pack_into( '<i', self.bytes, offset, value )

    def set_uint32( self, offset: int, value: int ):
        struct.pack_into( '<I', self.bytes, offset, value )


MODEL_N_GPU_LAYERS = 16
CONTEXT_N_CTX, CONTEXT_N_BATCH, CONTEXT_N_UBATCH = 0, 4, 8


def load_library( directory: str ):
    os.add_dll_directory( directory )

    # The release builds its CPU and CUDA backends as plugins; they load from the directory before any model does.
    ggml = ctypes.CDLL( os.path.join( directory, 'ggml.dll' ) )
    ggml.ggml_backend_load_all_from_path.argtypes = [ ctypes.c_char_p ]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path( directory.encode() )

    llama = ctypes.CDLL( os.path.join( directory, 'llama.dll' ) )

    llama.llama_backend_init.restype = None
    llama.llama_model_default_params.restype = Parameters
    llama.llama_context_default_params.restype = Parameters
    llama.llama_model_load_from_file.argtypes = [ ctypes.c_char_p, Parameters ]
    llama.llama_model_load_from_file.restype = ctypes.c_void_p
    llama.llama_init_from_model.argtypes = [ ctypes.c_void_p, Parameters ]
    llama.llama_init_from_model.restype = ctypes.c_void_p
    llama.llama_model_get_vocab.argtypes = [ ctypes.c_void_p ]
    llama.llama_model_get_vocab.restype = ctypes.c_void_p
    llama.llama_vocab_n_tokens.argtypes = [ ctypes.c_void_p ]
    llama.llama_vocab_n_tokens.restype = ctypes.c_int32
    llama.llama_vocab_bos.argtypes = [ ctypes.c_void_p ]
    llama.llama_vocab_bos.restype = ctypes.c_int32
    llama.llama_tokenize.argtypes = [ ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                      ctypes.POINTER( ctypes.c_int32 ), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool ]
    llama.llama_tokenize.restype = ctypes.c_int32
    llama.llama_batch_init.argtypes = [ ctypes.c_int32, ctypes.c_int32, ctypes.c_int32 ]
    llama.llama_batch_init.restype = Batch
    llama.llama_decode.argtypes = [ ctypes.c_void_p, Batch ]
    llama.llama_decode.restype = ctypes.c_int32
    llama.llama_get_logits.argtypes = [ ctypes.c_void_p ]
    llama.llama_get_logits.restype = ctypes.POINTER( ctypes.c_float )

    return llama


def tokenize( llama, vocab, text: str ) -> list[ int ]:
    data = text.encode( 'utf-8' )
    capacity = len( data ) + 16
    tokens = ( ctypes.c_int32 * capacity )()
    count = llama.llama_tokenize( vocab, data, len( data ), tokens, capacity, False, True )

    if count < 0:
        raise RuntimeError( f'llama_tokenize needs {-count} tokens' )

    return list( tokens[ :count ] )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--llama', required=True, help='directory holding llama.dll' )
    parser.add_argument( '--model', required=True )
    parser.add_argument( '--book', required=True )
    parser.add_argument( '--tokens', type=int, default=32768 )
    parser.add_argument( '--family', choices=[ 'gemma', 'llama' ], default='gemma' )
    parser.add_argument( '--prefixes', type=int, nargs='+', help='band ends; 8K bands when omitted' )
    arguments = parser.parse_args()

    book_turn = GEMMA_BOOK_TURN if arguments.family == 'gemma' else LLAMA_BOOK_TURN

    llama = load_library( arguments.llama )
    llama.llama_backend_init()

    model_parameters = llama.llama_model_default_params()
    model_parameters.set_int32( MODEL_N_GPU_LAYERS, 999 )
    model = llama.llama_model_load_from_file( arguments.model.encode(), model_parameters )

    if not model:
        raise RuntimeError( f'could not load {arguments.model}' )

    vocab = llama.llama_model_get_vocab( model )
    vocabulary = llama.llama_vocab_n_tokens( vocab )

    with open( arguments.book, 'rb' ) as book:
        text = join_wraps( book.read( arguments.tokens * 6 ).decode( 'utf-8', errors='replace' ) )

    prompt = [ llama.llama_vocab_bos( vocab ) ] + tokenize( llama, vocab, book_turn )
    ids = ( prompt + tokenize( llama, vocab, text ) )[ : arguments.tokens ]

    print( f'  prompt ids: {" ".join( str( id ) for id in prompt )}' )
    print( f'  {len( ids )} tokens, {len( prompt )} of prompt, vocabulary {vocabulary}', flush=True )

    context_parameters = llama.llama_context_default_params()
    context_parameters.set_uint32( CONTEXT_N_CTX, len( ids ) )
    context_parameters.set_uint32( CONTEXT_N_BATCH, CHUNK )
    context_parameters.set_uint32( CONTEXT_N_UBATCH, CHUNK )
    context = llama.llama_init_from_model( model, context_parameters )

    if not context:
        raise RuntimeError( 'could not create a context' )

    batch = llama.llama_batch_init( CHUNK, 0, 1 )
    log_probabilities = np.empty( len( ids ) - 1, dtype=np.float64 )
    largest_logit = 0.0

    for start in range( 0, len( ids ), CHUNK ):
        chunk = ids[ start : start + CHUNK ]
        batch.n_tokens = len( chunk )

        for row, token in enumerate( chunk ):
            batch.token[ row ] = token
            batch.pos[ row ] = start + row
            batch.n_seq_id[ row ] = 1
            batch.seq_id[ row ][ 0 ] = 0
            batch.logits[ row ] = 1

        status = llama.llama_decode( context, batch )

        if status != 0:
            raise RuntimeError( f'llama_decode returned {status} at {start}' )

        logits = np.ctypeslib.as_array( llama.llama_get_logits( context ), shape=( len( chunk ), vocabulary ) )

        # Row r predicts the token at start + r + 1; the last row of the last chunk predicts nothing.
        rows = min( len( chunk ), len( ids ) - 1 - start )
        targets = np.asarray( ids[ start + 1 : start + 1 + rows ] )
        values = logits[ :rows ].astype( np.float64 )
        largest_logit = max( largest_logit, float( np.abs( values ).max() ) )
        peaks = values.max( axis=1, keepdims=True )
        normalizers = peaks[ :, 0 ] + np.log( np.exp( values - peaks ).sum( axis=1 ) )
        log_probabilities[ start : start + rows ] = values[ np.arange( rows ), targets ] - normalizers

        if ( start + CHUNK ) % BAND == 0:
            print( f'  scored to {start + CHUNK}', flush=True )

    # A softcapped head never exceeds the cap (30 for Gemma); a larger logit would mean the cap was not applied.
    print( f'  largest |logit| {largest_logit:.3f}' )

    first = len( prompt ) - 1
    ends = arguments.prefixes or list( range( BAND, len( ids ) + BAND, BAND ) )

    for band_start, band_end in zip( [ 0 ] + ends[ :-1 ], ends ):
        low = max( band_start - 1, first )
        high = min( band_end, len( ids ) ) - 1
        band = log_probabilities[ low : high ]
        print( f'  {band_start:>7} - {band_end:>7}: {band.size:>5} positions, {-band.mean():.4f} nats/token' )


if __name__ == '__main__':
    main()
