# Per-layer hidden-state summary for Gemma 4 parity LOCALIZATION (companion to
# hf_gemma_greedy_validation.py). Runs ONE forward pass over the same prompt ids the
# parity test feeds (kPromptIds) and PRINTS a one-line summary of each decoder layer's
# LAST-TOKEN hidden state, in the same format Mila prints (Gemma.ixx kGemmaDumpActivations):
#
#   [HF-DUMP] layer_05    l2=... mean=... min=... max=... head=[.., .., ..]
#
# Read the two consoles side by side, top-down. Matching numbers = that stage agrees;
# the first stage whose l2/min/max diverges by orders of magnitude is where Mila breaks.
# Uses forward hooks so each captured tensor is unambiguously that layer's output.
#
# REQUIRES CUDA + bfloat16 to match Mila's compute precision; run it where you ran
# hf_gemma_greedy_validation.py (12B bf16 needs ~24 GB).

import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-12b-it"

if not torch.cuda.is_available():
    print( "ERROR: CUDA is required (bf16 on CPU is numerically unstable for 48 layers)." )
    sys.exit( 1 )

tokenizer = AutoTokenizer.from_pretrained( MODEL_ID )
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager" ).to( "cuda" ).eval()

# Identical prompt construction to hf_gemma_greedy_validation.py -> ids == kPromptIds.
messages = [ { "role": "user", "content": "The capital of France is" } ]
prompt_text = tokenizer.apply_chat_template( messages, add_generation_prompt=True, tokenize=False )
token_ids = tokenizer.encode( prompt_text, add_special_tokens=False )
print( "prompt ids:", token_ids, "(check these match kPromptIds)\n" )
ids = torch.tensor( [ token_ids ], dtype=torch.long, device="cuda" )

text_model = model.model
if hasattr( text_model, "language_model" ):
    text_model = text_model.language_model
layers, final_norm = text_model.layers, text_model.norm

# DEFINITIVE Gemma 4 architecture (do NOT assume Gemma 3): print the EXACT source of the
# RMSNorm, decoder layer, MLP, and attention forward methods from the LOADED model. This
# resolves (a) the real RMSNorm weight convention -- Gemma3's (1+weight) one-hots this model's
# attention, so Gemma 4's QK-norm is effectively raw -- and (b) the residual/FFN structure that
# the standard-sandwich oracle does NOT reproduce (Mila/oracle res2 ~1640 vs HF 88). Read these
# and diff against Mila's GemmaBlock before touching norms again.
import inspect
for label, obj in [
    ( "RMSNorm.forward",        type( layers[ 0 ].input_layernorm ) ),
    ( "DecoderLayer.forward",   type( layers[ 0 ] ) ),
    ( "MLP.forward",            type( layers[ 0 ].mlp ) ),
    ( "Attention.forward",      type( layers[ 0 ].self_attn ) ),
]:
    print( f"\n===== {type(layers[0]).__name__} :: {label} ({obj.__name__}) =====" )
    try:
        print( inspect.getsource( obj.forward ) )
    except Exception as e:
        print( f"(getsource failed: {e})" )
try:
    print( "\n===== RMSNorm._norm =====" )
    print( inspect.getsource( type( layers[ 0 ].input_layernorm )._norm ) )
except Exception:
    pass
print( "config.architectures:", getattr( model.config, "architectures", None ) )

# layer_scalar (the missing per-layer output multiply), per-norm with_scale flags, and v_norm.
def _scalar( x ):
    try:
        return float( x.reshape( -1 )[ 0 ] ) if hasattr( x, "reshape" ) else float( x )
    except Exception:
        return x
print( "layer_scalar by layer:",
       { i: _scalar( getattr( layers[ i ], "layer_scalar", None ) ) for i in [ 0, 1, 5, 11, 23, 47 ] } )
print( "layer_scalar type:", type( getattr( layers[ 0 ], "layer_scalar", None ) ),
       "shape:", getattr( getattr( layers[ 0 ], "layer_scalar", None ), "shape", None ) )
# Per-layer-type norm flags: layer 0 (local) vs 5/11 (global anchors). with_scale and v_proj
# (None => global K=V), layer_type, is_kv_shared_layer all may differ local vs global.
for li in [ 0, 1, 5, 11 ]:
    if li >= len( layers ):
        continue
    lyr = layers[ li ]
    a = lyr.self_attn
    print( f"--- layer {li} (layer_type={getattr(a,'layer_type',None)} "
           f"is_kv_shared={getattr(a,'is_kv_shared_layer',None)} v_proj={'None' if getattr(a,'v_proj',None) is None else 'present'} "
           f"sliding_window={getattr(a,'sliding_window',None)}) ---" )
    for nm in [ "q_norm", "k_norm", "v_norm" ]:
        n = getattr( a, nm, None )
        wl2 = float( n.weight.float().norm() ) if ( n is not None and getattr( n, "weight", None ) is not None ) else None
        print( f"  attn.{nm}: exists={n is not None} with_scale={getattr(n,'with_scale',None)} weight_l2={wl2}" )
    for nm in [ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" ]:
        n = getattr( lyr, nm, None )
        print( f"  layer.{nm}: with_scale={getattr(n,'with_scale',None)}" )
print( "final norm.with_scale:", getattr( final_norm, "with_scale", None ) )

captured = {}

def layer_hook( index ):
    def hook( module, inputs, output ):
        captured[ f"layer_{index:02d}" ] = ( output[ 0 ] if isinstance( output, tuple ) else output ).detach()
        if index == 0:
            captured[ "embed" ] = inputs[ 0 ].detach()   # scaled embeddings = input to layer 0
    return hook

handles = [ layers[ i ].register_forward_hook( layer_hook( i ) ) for i in range( len( layers ) ) ]
handles.append( final_norm.register_forward_hook( lambda m, i, o: captured.__setitem__( "final_norm", o.detach() ) ) )

# Layer-0 SUB-STEP hooks, to compare against Mila's [BLOCK-DUMP] tf_layer_0 chain.
l0 = layers[ 0 ]
def sub_hook( key ):
    def hook( module, inputs, output ):
        captured[ key ] = ( output[ 0 ] if isinstance( output, tuple ) else output ).detach()
    return hook
for attr, key in [ ( "input_layernorm", "blk_input_norm" ),
                   ( "self_attn", "blk_attn" ),
                   ( "post_attention_layernorm", "blk_post_attn_norm" ),
                   ( "pre_feedforward_layernorm", "blk_pre_ffn_norm" ),
                   ( "mlp", "blk_mlp" ),
                   ( "post_feedforward_layernorm", "blk_post_ffn_norm" ) ]:
    if hasattr( l0, attr ):
        handles.append( getattr( l0, attr ).register_forward_hook( sub_hook( key ) ) )

# Attention INTERNALS: q_norm / k_norm outputs + the pre-o_proj attention (o_proj input).
attn0 = l0.self_attn
print( "self_attn.scaling:", getattr( attn0, "scaling", None ),
       " query_pre_attn_scalar:", getattr( model.config, "query_pre_attn_scalar", None ),
       " head_dim:", getattr( model.config, "head_dim", None ) )
if hasattr( attn0, "q_norm" ):
    handles.append( attn0.q_norm.register_forward_hook( sub_hook( "blk_q_norm" ) ) )
if hasattr( attn0, "k_norm" ):
    handles.append( attn0.k_norm.register_forward_hook( sub_hook( "blk_k_norm" ) ) )
if hasattr( attn0, "v_proj" ):
    handles.append( attn0.v_proj.register_forward_hook( sub_hook( "blk_v_proj" ) ) )
handles.append( attn0.o_proj.register_forward_pre_hook(
    lambda m, args: captured.__setitem__( "blk_attn_preo", args[ 0 ].detach() ) ) )

# res1 (input + post_attn_norm(attn)) is the INPUT to pre_feedforward_layernorm. HF never
# captures it directly, so pre_ffn_norm was the first comparable point past the attention
# residual -- capture res1 here to localize whether the divergence is in the residual/attention
# (res1 differs) or in the norm itself (res1 matches but pre_ffn_norm does not).
handles.append( l0.pre_feedforward_layernorm.register_forward_pre_hook(
    lambda m, args: captured.__setitem__( "blk_res1", args[ 0 ].detach() ) ) )

# ffn (= mlp output = fc_down output) as a GENUINE ACTIVATION tensor via a pre-hook on
# post_feedforward_layernorm -- reliable, unlike the post_ffn_norm FORWARD hook (norm-module
# outputs over-capture: post_ffn_norm 1595 (+) res1 333 is incompatible with layer_00 88).
handles.append( l0.post_feedforward_layernorm.register_forward_pre_hook(
    lambda m, args: captured.__setitem__( "blk_ffn_reliable", args[ 0 ].detach() ) ) )

# FFN-body GENUINE ACTIVATIONS (reliable hooks) to split GeGLU/fc_gate_up from fc_down:
#   gate_proj/up_proj outputs, and the GeGLU/act output (= down_proj INPUT, pre-hook).
# Compare HF act-output to Mila `geglu` (1448) and HF gate/up to Mila's gate_up halves.
handles.append( l0.mlp.gate_proj.register_forward_hook(
    lambda m, i, o: captured.__setitem__( "blk_gate_proj", o.detach() ) ) )
handles.append( l0.mlp.up_proj.register_forward_hook(
    lambda m, i, o: captured.__setitem__( "blk_up_proj", o.detach() ) ) )
handles.append( l0.mlp.down_proj.register_forward_pre_hook(
    lambda m, args: captured.__setitem__( "blk_geglu", args[ 0 ].detach() ) ) )

# layer-0 TRUE OUTPUT residual, captured independently as layer-1's INPUT (a genuine residual
# tensor) -- cross-checks the [HF-DUMP] layer_00 value (88). If both read 88, the post_ffn_norm
# FORWARD hook (1595) is definitively bogus and HF's FFN nets a small/canceling contribution.
if len( layers ) > 1:
    handles.append( layers[ 1 ].register_forward_pre_hook(
        lambda m, args: captured.__setitem__( "res_layer0_out", args[ 0 ].detach() ) ) )

with torch.no_grad():
    out = model( ids, use_cache=False, output_attentions=True, output_hidden_states=True )

# CANONICAL residual stream via output_hidden_states (HF-internal, NOT a manual forward hook).
# out.hidden_states = (embed, layer0_out, layer1_out, ...). This is the reliable arbiter:
#   layer0_out ~= 88  -> the standard-math oracle (1643) is WRONG -> Gemma has a non-standard
#                        layer op both the oracle and Mila miss (case B).
#   layer0_out ~= 1640 -> the forward-hook trajectory (88) was ALSO bogus; Mila's layers match
#                        the real residual -> the bug is DOWNSTREAM (final_norm/lm_head/sampling,
#                        case A). Compare each to Mila [GEMMA-DUMP] layer_NN l2.
if getattr( out, "hidden_states", None ) is not None:
    print()
    for i, hs in enumerate( out.hidden_states ):
        v = hs[ 0, -1, : ].to( torch.float32 ).cpu().numpy().reshape( -1 )
        l2 = float( math.sqrt( float( ( v * v ).sum() ) ) )
        tag = "embed" if i == 0 else f"layer_{i-1:02d}"
        print( f"[HF-HS] {tag:<10} l2={l2:>12.4f} min={v.min():>+10.4f} max={v.max():>+10.4f}" )

# Layer-0 per-head last-query attention distribution -- compare to Mila [GQA-DUMP] att(hN).
# If HF heads are MUCH softer than Mila (lower max) -> Mila scores too sharp -> RoPE/QK bug;
# if HF matches Mila's argmax/max -> the pattern is right and it's the V/output magnitude.
if out.attentions is not None:
    print()
    nh = out.attentions[ 0 ].shape[ 1 ]
    # FULL last-query att distribution for ALL heads, first 2 LOCAL blocks (0,1) + first GLOBAL
    # block (5, the 5:1 pattern) -- compare element-wise to Mila [GQA-ROW] L{0,1,5} att(h*).
    # A per-head row that diverges (esp. a key/position swap or a sharper/softer head) localizes
    # the attention bug; if every head's row matches but the output direction is still wrong, the
    # culprit is V-per-channel or the att*V/unpermute combine, not the scores.
    for layer_idx in [ 0, 1, 5 ]:
        if layer_idx >= len( out.attentions ):
            continue
        att_l = out.attentions[ layer_idx ][ 0 ]   # [nh, q, k]
        nh_l = att_l.shape[ 0 ]
        print( f"[HF-ATTN] layer {layer_idx} ({nh_l} heads)" )
        for h in range( nh_l ):
            a = att_l[ h, -1, : ].to( torch.float32 ).cpu().numpy()
            print( f"[HF-ATTN-ROW] L{layer_idx} h{h:<2}:" + "".join( f" {x:.4f}" for x in a ) )

# ============================================================================
# GROUND-TRUTH ORACLE: att*V -> o_proj -> post_attn_norm for layer 0, LAST query.
# Computed in fp32 numpy from RELIABLE inputs only -- the softmax weights
# (output_attentions, which match Mila) and the v_proj activation -- so it
# BYPASSES the magnitude-bogus attn/o_proj forward hooks. This is the true
# attention output. Compare to Mila [BLOCK-T17] attn/o_proj/post_attn_norm:
#   - oracle == Mila  -> att*V/o_proj/unpermute are correct; chase V-per-channel
#                        upstream (the v_proj activation itself).
#   - oracle != Mila  -> the att*V combine / o_proj / unpermute scrambles the
#                        output even though att and V match -> that is the bug.
# Also sanity-checks HF's own pipeline: oracle post_attn_norm should ~= the HF
# post_attn_norm hook direction (the reliable one, == res1 - embed).
import numpy as np

if out.attentions is not None and "blk_v_proj" in captured:
    text_cfg = getattr( model.config, "text_config", model.config )
    nh_o  = out.attentions[ 0 ].shape[ 1 ]
    nkv_o = int( getattr( text_cfg, "num_key_value_heads", 8 ) )
    hs_o  = int( getattr( text_cfg, "head_dim", 256 ) )
    gs_o  = nh_o // nkv_o
    eps_o = float( getattr( text_cfg, "rms_norm_eps", 1e-6 ) )

    att0 = out.attentions[ 0 ][ 0, :, -1, : ].to( torch.float32 ).cpu().numpy()   # [nh, k]
    Vp   = captured[ "blk_v_proj" ][ 0 ].to( torch.float32 ).cpu().numpy()        # [k, nkv*hs]
    k_len = Vp.shape[ 0 ]
    Vp = Vp.reshape( k_len, nkv_o, hs_o )

    attn_out = np.zeros( ( nh_o, hs_o ), dtype=np.float64 )
    for h in range( nh_o ):
        kv = h // gs_o                          # HF repeat_kv: head h uses kv head h//GS
        attn_out[ h ] = ( att0[ h ][ :, None ] * Vp[ :, kv, : ] ).sum( axis=0 )
    attn_flat = attn_out.reshape( -1 )          # [nh*hs], head-major (o_proj input order)

    Wo = l0.self_attn.o_proj.weight.detach().to( torch.float32 ).cpu().numpy()    # [out, in]
    o  = Wo @ attn_flat

    w_post = l0.post_attention_layernorm.weight.detach().to( torch.float32 ).cpu().numpy()
    rms = float( np.sqrt( ( o * o ).mean() + eps_o ) )
    pan = ( o / rms ) * ( 1.0 + w_post )        # Gemma3 RMSNorm: x_norm * (1 + weight) -- per HF source

    def ostat( label, v ):
        v = np.asarray( v ).reshape( -1 )
        l2 = float( np.sqrt( float( ( v * v ).sum() ) ) )
        print( f"[ORACLE] {label:<16} n={v.size:>5} l2={l2:>12.4f} "
               f"head=[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}, {v[3]:+.4f}]" )

    print()
    print( f"[ORACLE] L0 nh={nh_o} nkv={nkv_o} hs={hs_o} gs={gs_o} eps={eps_o}" )
    ostat( "attn", attn_flat )
    ostat( "o_proj", o )
    ostat( "post_attn_norm", pan )

    # ---- Continue the SAME oracle through the FFN to res2, all fp32 from HF weights.
    # res1 = embed + post_attn (oracle attention output). If the oracle res2 ~= the reliable
    # HF layer-0 output (88) but Mila res2 = 1634, the FFN is the bug -- and the per-step
    # oracle vs Mila [BLOCK-T17] (pre_ffn_norm/geglu/fc_down/post_ffn_norm) pinpoints the op.
    if "embed" in captured:
        def rmsnorm( x, w ):
            return ( x / np.sqrt( ( x * x ).mean() + eps_o ) ) * ( 1.0 + w )   # Gemma3: x_norm*(1+weight)

        embed_last = captured[ "embed" ][ 0, -1, : ].to( torch.float32 ).cpu().numpy()
        res1 = embed_last + pan

        w_pre  = l0.pre_feedforward_layernorm.weight.detach().to( torch.float32 ).cpu().numpy()
        Wg     = l0.mlp.gate_proj.weight.detach().to( torch.float32 ).cpu().numpy()
        Wu     = l0.mlp.up_proj.weight.detach().to( torch.float32 ).cpu().numpy()
        Wd     = l0.mlp.down_proj.weight.detach().to( torch.float32 ).cpu().numpy()
        w_post = l0.post_feedforward_layernorm.weight.detach().to( torch.float32 ).cpu().numpy()

        ffn_in = rmsnorm( res1, w_pre )
        gate   = Wg @ ffn_in
        up     = Wu @ ffn_in
        gelu   = 0.5 * gate * ( 1.0 + np.tanh( np.sqrt( 2.0 / np.pi ) * ( gate + 0.044715 * gate ** 3 ) ) )
        geglu  = gelu * up
        fcd    = Wd @ geglu
        post_ffn = rmsnorm( fcd, w_post )
        res2   = res1 + post_ffn

        ostat( "res1", res1 )
        ostat( "pre_ffn_norm", ffn_in )
        ostat( "geglu", geglu )
        ostat( "fc_down", fcd )
        ostat( "post_ffn_norm", post_ffn )
        ostat( "res2", res2 )   # compare to HF reliable layer-0 output (88) and Mila (1634)

# Token-0 (BOS) DIRECTION trace for the V0 sink bug -- compare head values to Mila [BLOCK-T0].
def t0( label, vec ):
    v = vec.to( torch.float32 ).cpu().numpy().reshape( -1 )
    l2 = float( math.sqrt( float( ( v * v ).sum() ) ) )
    print( f"[HF-T0] {label:<16} n={v.size:>5} l2={l2:>12.4f} "
           f"head=[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}, {v[3]:+.4f}]" )

t0( "input_norm(t0)", captured[ "blk_input_norm" ][ 0, 0, : ] )
t0( "input_norm(tL)", captured[ "blk_input_norm" ][ 0, -1, : ] )
if "blk_v_proj" in captured:
    t0( "V(t0)", captured[ "blk_v_proj" ][ 0, 0, : ] )
    t0( "V(tL)", captured[ "blk_v_proj" ][ 0, -1, : ] )
    # V per KV head, last token -- compare ORDER to Mila [V-KVHEAD].
    vL = captured[ "blk_v_proj" ][ 0, -1, : ].to( torch.float32 ).cpu().numpy()
    HD = 256
    parts = " ".join( f"kv{k}={math.sqrt(float((vL[k*HD:(k+1)*HD]**2).sum())):.2f}"
                      for k in range( vL.size // HD ) )
    print( "[HF-V-KVHEAD] last-token l2 per kv:", parts )

for h in handles:
    h.remove()

def summary( label, tensor ):
    row = tensor[ 0, -1, : ].to( torch.float32 ).cpu().numpy()   # last-token hidden vector
    l2 = float( math.sqrt( float( ( row * row ).sum() ) ) )
    print( f"[HF-DUMP] {label:<11} l2={l2:>11.4f} mean={row.mean():>+10.5f} "
           f"min={row.min():>+10.4f} max={row.max():>+10.4f} "
           f"head=[{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]" )

summary( "embed", captured[ "embed" ] )
for i in range( len( layers ) ):
    summary( f"layer_{i:02d}", captured[ f"layer_{i:02d}" ] )
summary( "final_norm", captured[ "final_norm" ] )

# ---- Layer-0 sub-step ground truth (compare to Mila [BLOCK-DUMP] tf_layer_0) ----
print()

def vstats( label, vec ):
    v = vec.to( torch.float32 ).cpu().numpy().reshape( -1 )
    l2 = float( math.sqrt( float( ( v * v ).sum() ) ) )
    h = v[ :4 ]
    print( f"[HF-BLOCK] {label:<16} n={v.size:>6} l2={l2:>12.4f} min={v.min():>+12.4f} max={v.max():>+12.4f}"
           f" head=[{h[0]:+.4f}, {h[1]:+.4f}, {h[2]:+.4f}, {h[3]:+.4f}]" )

# Gemma 4 RMSNorm applies the RAW stored weight (x_norm * weight), NOT the documented
# (1 + weight) shift -- proven by the input_norm token-0 trace (Mila raw == HF effective).
# So print the RAW weights; Mila's loaded .W (also raw) must match these element-wise.
vstats( "input_norm.W_raw",     l0.input_layernorm.weight.detach() )
vstats( "q_norm.W_raw",         attn0.q_norm.weight.detach() )
vstats( "k_norm.W_raw",         attn0.k_norm.weight.detach() )
# The three sandwich norms NOT yet element-wise verified after the raw-weight re-convert.
# If Mila's loaded post_ffn_norm.W (RMS/min/max) differs from this raw weight, the FFN
# residual inflation (layer-0 res2 1634 vs HF 88) is a weight-LOAD bug; if they match, it
# is the FFN body (fc_gate_up/fc_down direction) -> compare ffn_reliable below.
vstats( "post_attn_norm.W_raw", l0.post_attention_layernorm.weight.detach() )
vstats( "pre_ffn_norm.W_raw",   l0.pre_feedforward_layernorm.weight.detach() )
vstats( "post_ffn_norm.W_raw",  l0.post_feedforward_layernorm.weight.detach() )
for label, key in [ ( "input_norm", "blk_input_norm" ),
                    ( "q_norm", "blk_q_norm" ), ( "k_norm", "blk_k_norm" ),
                    ( "attn_pre_oproj", "blk_attn_preo" ), ( "attn", "blk_attn" ),
                    ( "post_attn_norm", "blk_post_attn_norm" ), ( "res1(+attn)", "blk_res1" ),
                    ( "pre_ffn_norm", "blk_pre_ffn_norm" ),
                    ( "gate_proj", "blk_gate_proj" ), ( "up_proj", "blk_up_proj" ),
                    ( "geglu", "blk_geglu" ),
                    ( "mlp", "blk_mlp" ), ( "ffn_reliable", "blk_ffn_reliable" ),
                    ( "post_ffn_norm", "blk_post_ffn_norm" ),
                    ( "res2=layer0_out", "res_layer0_out" ) ]:
    if key in captured:
        vstats( label, captured[ key ][ 0, -1, : ] )
