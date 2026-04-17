#pragma once
// pipeline-synth-impl.h: private definition of AceSynth and AceSynthJob
//
// AceSynth holds the lightweight modules that stay resident across jobs
// (TextEnc, CondEnc, FSQ tok/detok, BPE) plus CPU-side DiT metadata and
// the null condition embedding.
// DiT and VAE decoder are loaded on demand via explicit phase calls and may
// be absent at any given moment.
//
// AceSynthJob carries the per-request state between phase 1 (DiT) and
// phase 2 (VAE). The orchestrator produces one job per ace_synth_job_run_dit
// call, then feeds it back to ace_synth_job_run_vae later.
//
// This header is included by the two implementation files:
//   pipeline-synth.cpp      orchestrator: load, phases, free
//   pipeline-synth-ops.cpp  primitives: encode, context, noise, dit, vae

#include "bpe.h"
#include "cond-enc.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "pipeline-synth-ops.h"
#include "pipeline-synth.h"
#include "qwen3-enc.h"
#include "vae.h"

#include <vector>

struct AceSynth {
    // Resident modules (loaded once at ace_synth_load)
    Qwen3GGML    text_enc;
    CondGGML     cond_enc;
    DetokGGML    detok;
    TokGGML      tok;
    BPETokenizer bpe;

    // On-demand modules (loaded via ace_synth_dit_load / ace_synth_vae_load)
    DiTGGML dit;
    VAEGGML vae;
    bool    have_dit;  // true when dit carries GPU weights
    bool    have_vae;  // true when vae carries GPU weights

    // CPU-side DiT metadata, populated at ace_synth_load from the GGUF.
    // Allows text encoding and T resolution to run without the DiT in VRAM.
    DiTGGMLConfig      dit_cfg;
    std::vector<float> silence_full;   // [15000, 64] f32, from silence_latent tensor
    std::vector<float> null_cond_cpu;  // [hidden_size] f32, from null_condition_emb (empty when model has none)
    bool               is_turbo;

    // Config
    AceSynthParams params;
    bool           have_detok;
    bool           have_tok;

    // Derived constants
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)
};

// Job carries phase 1 outputs across to phase 2.
// The latent tensor is held in state.output: planar [batch_n, Oc, T] f32.
struct AceSynthJob {
    SynthState state;
    int        batch_n;
};
