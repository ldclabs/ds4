// ds4 — DeepSeek V4 Flash inference CLI (Rust port of ds4_cli.c)
//
// Usage:
//   ds4 -m MODEL.gguf -p "Hello, world!"
//   ds4 -m MODEL.gguf                         (interactive chat)
//   ds4 -m MODEL.gguf --chat-template deepseek (default: auto-detect)

use std::env;
use std::io::{self, BufRead, Write};
use std::process;
use std::time::Instant;

use ds4::gguf::GgufModel;
use ds4::model::bind_weights_unchecked;
use ds4::tokenizer::Vocab;
use ds4::session::Session;
use ds4::constants::*;

struct Config {
    model_path: String,
    prompt: Option<String>,
    system_prompt: Option<String>,
    prompt_file: Option<String>,
    n_predict: usize,
    ctx_size: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    think_mode: ThinkMode,
    quiet: bool,
    debug_tokens: bool,
    batched: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum ThinkMode {
    Auto,
    Normal,
    Max,
    Off,
}

fn main() {
    let cfg = parse_args();
    println!("Loading model: {}", cfg.model_path);

    let gguf = GgufModel::open(&cfg.model_path).unwrap_or_else(|e| {
        eprintln!("Failed to open GGUF: {}", e);
        process::exit(1);
    });
    println!("  version={}, tensors={}", gguf.version, gguf.tensors.len());

    let weights = bind_weights_unchecked(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to bind weights: {}", e);
        process::exit(1);
    });
    println!("  layers={}, n_embd={}", weights.layers.len(), N_EMBD);

    let vocab = match Vocab::load(&gguf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Warning: Failed to load tokenizer: {}", e);
            eprintln!("  One-shot mode requires a valid tokenizer. Exiting.");
            process::exit(1);
        }
    };
    println!("  vocab_size={}, bos={}, eos={}", vocab.n_vocab, vocab.bos_id, vocab.eos_id);

    // Build prompt
    let prompt_text = match (&cfg.prompt, &cfg.prompt_file) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => {
            std::fs::read_to_string(f).unwrap_or_else(|e| {
                eprintln!("Failed to read prompt file: {}", e);
                process::exit(1);
            })
        }
        (None, None) => {
            // Interactive mode
            run_interactive(&weights, &vocab, &cfg);
            return;
        }
    };

    // Apply chat template
    let prompt_tokens = build_chat_prompt(&[], &prompt_text, cfg.system_prompt.as_deref(), cfg.think_mode, &vocab);

    if !cfg.quiet {
        println!("\n--- prompt ---\n{}", prompt_tokens_display(&vocab, &prompt_tokens));
        println!("--- encoding ---");
    }

    if !cfg.quiet {
        println!("  encoded {} tokens", prompt_tokens.len());
    }

    // Run generation
    run_oneshot(&weights, &vocab, &cfg, &prompt_tokens);
}

fn run_oneshot(weights: &ds4::model::ModelWeights, vocab: &Vocab, cfg: &Config, prompt_tokens: &[i32]) {
    let mut session = Session::new(cfg.ctx_size);
    let mut rng = cfg.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    // Sync prompt
    if !cfg.quiet {
        eprintln!("prefill: processing {} tokens through {} layers...", prompt_tokens.len(), weights.layers.len());
    }
    let start = Instant::now();
    session.sync(weights, prompt_tokens, cfg.batched);
    let prefill_time = start.elapsed();
    if !cfg.quiet {
        eprintln!("prefill: {:?} ({:.1} ms) — {} tokens",
            prefill_time, prefill_time.as_secs_f64() * 1000.0, prompt_tokens.len());
    }

    // NaN check on logits
    let nan_count = session.logits.iter().filter(|&&v| v.is_nan()).count();
    let inf_count = session.logits.iter().filter(|&&v| v.is_infinite()).count();
    let has_valid = session.logits.iter().any(|&v| v.is_finite());
    if !cfg.quiet {
        eprintln!("logits: {} NaN, {} Inf, has_finite={}", nan_count, inf_count, has_valid);
    }
    if cfg.debug_tokens {
        // Print top-5 tokens with logprobs
        let max_val = session.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut indexed: Vec<(usize, f32)> = session.logits.iter().enumerate()
            .map(|(i, &v)| (i, (v - max_val).exp()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
        eprintln!("top-5 logprobs after prefill:");
        for k in 0..5.min(indexed.len()) {
            let (id, prob) = indexed[k];
            let text = vocab.token_text_decoded(id as i32).unwrap_or_else(|| "<unk>".to_string());
            eprintln!("  [{:6}] {:>8.4}  {:?}", id, (prob/sum).ln(), text);
        }
    }
    if !has_valid {
        eprintln!("ERROR: all logits are NaN/Inf — model output is corrupted, aborting");
        return;
    }

    // Generate
    let mut generated = Vec::new();
    let spec_layers = ds4::forward::SPECULATIVE_DRAFT_LAYERS;
    let spec_count = ds4::forward::SPECULATIVE_DRAFT_TOKENS;
    let is_spec = cfg.temperature <= 0.001;
    let start = Instant::now();
    while generated.len() < cfg.n_predict {
        let token = if cfg.temperature <= 0.001 {
            session.argmax()
        } else {
            sample_token(&session.logits, cfg.temperature, cfg.top_p, &mut rng)
        };

        if token == vocab.eos_id {
            break;
        }

        // Decode and print token
        if cfg.debug_tokens {
            eprint!("[{}]", token);
        }
        if let Some(text) = vocab.token_text_decoded(token) {
            print!("{}", text);
            io::stdout().flush().ok();
        }

        // Feed base token + optionally draft/verify more via speculation
        if is_spec && generated.len() + spec_count + 1 <= cfg.n_predict {
            let (accepted, _) = session.eval_speculative(weights, token, spec_layers, spec_count);
            generated.push(token);
            for &draft in &accepted {
                if draft == vocab.eos_id { break; }
                if cfg.debug_tokens {
                    eprint!("[{}]", draft);
                }
                if let Some(text) = vocab.token_text_decoded(draft) {
                    print!("{}", text);
                    io::stdout().flush().ok();
                }
                generated.push(draft);
                if generated.len() >= cfg.n_predict { break; }
            }
        } else {
            session.eval(weights, token);
            generated.push(token);
        }
    }
    let gen_time = start.elapsed();
    let gen_tokens = generated.len();
    let tps = if gen_time.as_secs_f64() > 0.0 {
        gen_tokens as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    if !cfg.quiet && gen_tokens > 0 {
        eprintln!("\n\ngenerated {} tokens in {:?} ({:.1} tok/s)",
            gen_tokens, gen_time, tps);
    }
    println!();
}

fn run_interactive(weights: &ds4::model::ModelWeights, vocab: &Vocab, cfg: &Config) {
    println!("ds4> Interactive mode. Type /quit to exit, /help for commands.");
    println!("  Model loaded: {} layers, ctx_size={}", weights.layers.len(), cfg.ctx_size);

    let mut session = Session::new(cfg.ctx_size);
    let mut rng = cfg.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    let mut history: Vec<(String, String)> = Vec::new(); // (user, assistant) pairs

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        print!("\nds4> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("read error: {}", e);
                break;
            }
        }

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Commands
        if line.starts_with('/') {
            match line.as_str() {
                "/quit" | "/exit" => break,
                "/help" => {
                    println!("Commands:");
                    println!("  /quit, /exit  — exit");
                    println!("  /clear        — reset chat history");
                    println!("  /status       — show session stats");
                    println!("  /temp N       — set temperature (current: {:.2})", cfg.temperature);
                    continue;
                }
                "/clear" => {
                    history.clear();
                    session.invalidate();
                    println!("Chat history cleared.");
                    continue;
                }
                "/status" => {
                    println!("Session: {} tokens, {} history turns",
                        session.n_tokens(), history.len());
                    continue;
                }
                cmd if cmd.starts_with("/temp ") => {
                    // Parse doesn't actually change cfg since it's immutable,
                    // but we can report what they asked for
                    if let Ok(t) = cmd[6..].parse::<f32>() {
                        eprintln!("Temperature would change to {:.2} (restart with --temp to apply)", t);
                    }
                    continue;
                }
                _ => {
                    println!("Unknown command: {}. Type /help for commands.", line);
                    continue;
                }
            }
        }

        let user_input = line;

        // Build full prompt from history + current input
        let prompt_tokens = build_chat_prompt(&history, &user_input, cfg.system_prompt.as_deref(), cfg.think_mode, vocab);

        if !cfg.quiet {
            eprintln!("--- prompt ({} tokens) ---", prompt_tokens.len());
        }

        // Sync session to prompt
        if !cfg.quiet {
            eprintln!("prefill: processing {} tokens through {} layers...", prompt_tokens.len(), weights.layers.len());
        }
        let start = Instant::now();
        session.sync(weights, &prompt_tokens, cfg.batched);
        let prefill_time = start.elapsed();

        if !cfg.quiet {
            eprintln!("prefill: {:?} ({:.1} ms), {} tokens",
                prefill_time, prefill_time.as_secs_f64() * 1000.0, prompt_tokens.len());
        }

        // NaN check on logits
        let has_valid = session.logits.iter().any(|&v| v.is_finite());
        if !has_valid {
            eprintln!("ERROR: all logits are NaN/Inf — model output corrupted, try re-running");
            continue;
        }

        // Debug: print top-5 logprobs after prefill (chat mode)
        if cfg.debug_tokens {
            let max_val = session.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut indexed: Vec<(usize, f32)> = session.logits.iter().enumerate()
                .map(|(i, &v)| (i, (v - max_val).exp()))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
            eprintln!("top-5 logprobs after prefill:");
            for k in 0..5.min(indexed.len()) {
                let (id, prob) = indexed[k];
                let text = vocab.token_text_decoded(id as i32).unwrap_or_else(|| "<unk>".to_string());
                eprintln!("  [{:6}] {:>8.4}  {:?}", id, (prob/sum).ln(), text);
            }
            // Also print EOS logprob for reference
            let eos_logit = session.logits[vocab.eos_id as usize];
            eprintln!("  EOS(1) logit={:.4}", eos_logit);
        }

        // Generate assistant response
        let mut response = String::new();
        let gen_start = Instant::now();
        let mut gen_count = 0usize;

        let spec_layers = ds4::forward::SPECULATIVE_DRAFT_LAYERS;
        let spec_count = ds4::forward::SPECULATIVE_DRAFT_TOKENS;
        let is_spec = cfg.temperature <= 0.001;

        while gen_count < cfg.n_predict {
            let token = if cfg.temperature <= 0.001 {
                session.argmax()
            } else {
                sample_token(&session.logits, cfg.temperature, cfg.top_p, &mut rng)
            };

            if token == vocab.eos_id {
                if gen_count == 0 {
                    eprintln!("[DEBUG] first generated token is EOS — generation stopped");
                }
                break;
            }

            if cfg.debug_tokens {
                eprint!("[{}]", token);
            }
            if let Some(text) = vocab.token_text_decoded(token) {
                print!("{}", text);
                io::stdout().flush().ok();
                response.push_str(&text);
            }

            // Feed base token + optionally draft/verify more via speculation
            if is_spec && gen_count + spec_count + 1 <= cfg.n_predict {
                let (accepted, _) = session.eval_speculative(weights, token, spec_layers, spec_count);
                gen_count += 1;
                for &draft in &accepted {
                    if draft == vocab.eos_id { break; }
                    if cfg.debug_tokens {
                        eprint!("[{}]", draft);
                    }
                    if let Some(text) = vocab.token_text_decoded(draft) {
                        print!("{}", text);
                        io::stdout().flush().ok();
                        response.push_str(&text);
                    }
                    gen_count += 1;
                    if gen_count >= cfg.n_predict { break; }
                }
            } else {
                session.eval(weights, token);
                gen_count += 1;
            }
        }

        let gen_time = gen_start.elapsed();
        let tps = if gen_time.as_secs_f64() > 0.0 {
            gen_count as f64 / gen_time.as_secs_f64()
        } else {
            0.0
        };

        if !cfg.quiet && gen_count > 0 {
            eprintln!("\n[{} tokens in {:?}, {:.1} tok/s]", gen_count, gen_time, tps);
        }
        println!();

        history.push((user_input, response));
    }

    println!("\nGoodbye.");
}

/// Sample a token from logits with temperature and top-p.
fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng: &mut u64) -> i32 {
    let inv_temp = 1.0 / temperature.max(0.001);

    // Find max for numerical stability
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Apply temperature + softmax
    let mut probs: Vec<f32> = logits.iter()
        .map(|&v| {
            let scaled = (v - max_val) * inv_temp;
            if scaled > -50.0 { scaled.exp() } else { 0.0 }
        })
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        // Fallback: argmax
        let mut best = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val { best_val = v; best = i as i32; }
        }
        return best;
    }
    for p in &mut probs { *p /= sum; }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (k, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = k + 1;
                break;
            }
        }
        for i in cutoff..indexed.len() {
            probs[indexed[i].0] = 0.0;
        }

        // Renormalize
        let sum2: f32 = probs.iter().sum();
        if sum2 > 0.0 {
            for p in &mut probs { *p /= sum2; }
        }
    }

    // Xorshift RNG + sample
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    let r = (*rng as f64) / (u64::MAX as f64);

    let mut cumsum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p as f64;
        if r < cumsum {
            return i as i32;
        }
    }

    // Fallback
    probs.len() as i32 - 1
}

/// Build a DeepSeek chat prompt from history as token IDs.
/// Follows the C engine's convention: push special tokens by ID, tokenize
/// user/system text separately. This avoids the bug where Chinese words like
/// "比如"/"在校期间及" were mistaken for control markers.
fn build_chat_prompt(
    history: &[(String, String)],
    current_user: &str,
    system_prompt: Option<&str>,
    think_mode: ThinkMode,
    vocab: &Vocab,
) -> Vec<i32> {
    let mut tokens = Vec::new();

    // BOS
    tokens.push(vocab.bos_id);

    // System prompt (raw text, tokenized)
    let system = system_prompt.unwrap_or("You are a helpful assistant");
    tokens.extend(vocab.encode(system));

    // History
    for (user_msg, assistant_msg) in history {
        // User: <user_id> + tokenized text
        tokens.push(vocab.user_id);
        tokens.extend(vocab.encode(user_msg));

        // Assistant: <assistant_id> + think marker + tokenized response
        tokens.push(vocab.assistant_id);
        if think_mode != ThinkMode::Off {
            tokens.push(vocab.think_start_id);
        } else {
            tokens.push(vocab.think_end_id);
        }
        tokens.extend(vocab.encode(assistant_msg));
    }

    // Current user message
    tokens.push(vocab.user_id);
    tokens.extend(vocab.encode(current_user));

    // Assistant prefix (prompt the model to start responding)
    tokens.push(vocab.assistant_id);
    if think_mode != ThinkMode::Off {
        tokens.push(vocab.think_start_id);
    } else {
        tokens.push(vocab.think_end_id);
    }

    tokens
}

/// Convert prompt tokens to a human-readable display string for debug output.
/// Uses GPT-2 byte-decoded token text (spaces, newlines decode correctly).
fn prompt_tokens_display(vocab: &Vocab, tokens: &[i32]) -> String {
    tokens.iter()
        .map(|&id| vocab.token_text_decoded(id).unwrap_or_else(|| "<unk>".to_string()))
        .collect::<Vec<_>>()
        .join("")
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();

    let mut cfg = Config {
        model_path: "ds4flash.gguf".to_string(),
        prompt: None,
        system_prompt: None,
        prompt_file: None,
        n_predict: 50000,
        ctx_size: 32768,
        temperature: 0.7,
        top_p: 1.0,
        seed: None,
        think_mode: ThinkMode::Auto,
        quiet: false,
        debug_tokens: false,
        batched: true,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                if i < args.len() { cfg.model_path = args[i].clone(); }
            }
            "-p" | "--prompt" => {
                i += 1;
                if i < args.len() { cfg.prompt = Some(args[i].clone()); }
            }
            "--prompt-file" => {
                i += 1;
                if i < args.len() { cfg.prompt_file = Some(args[i].clone()); }
            }
            "-sys" | "--system" => {
                i += 1;
                if i < args.len() { cfg.system_prompt = Some(args[i].clone()); }
            }
            "-n" | "--tokens" => {
                i += 1;
                if i < args.len() {
                    cfg.n_predict = args[i].parse().unwrap_or(50000);
                }
            }
            "-c" | "--ctx" => {
                i += 1;
                if i < args.len() {
                    cfg.ctx_size = args[i].parse().unwrap_or(32768);
                }
            }
            "--temp" => {
                i += 1;
                if i < args.len() {
                    cfg.temperature = args[i].parse().unwrap_or(0.7);
                }
            }
            "--top-p" => {
                i += 1;
                if i < args.len() {
                    cfg.top_p = args[i].parse().unwrap_or(1.0);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    cfg.seed = Some(args[i].parse().unwrap_or(0));
                }
            }
            "--think" => cfg.think_mode = ThinkMode::Normal,
            "--think-max" => cfg.think_mode = ThinkMode::Max,
            "--no-think" => cfg.think_mode = ThinkMode::Off,
            "--quiet" | "-q" => cfg.quiet = true,
            "--debug-tokens" => cfg.debug_tokens = true,
            "--no-batched" => cfg.batched = false,
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown flag: {}", other);
                }
            }
        }
        i += 1;
    }

    cfg
}

fn print_usage() {
    println!(
        "Usage: ds4 [OPTIONS]

Invocation modes:
  ds4                          Interactive chat
  ds4 -p \"prompt\"              One-shot generation
  ds4 --prompt-file FILE       One-shot from file

Model:
  -m, --model FILE             GGUF model path (default: ds4flash.gguf)
  -c, --ctx N                  Context size (default: 32768)

Generation:
  -p, --prompt TEXT            Prompt text
  --prompt-file FILE           Read prompt from file
  -sys, --system TEXT          System prompt
  -n, --tokens N               Max tokens to generate (default: 50000)
  --temp F                     Temperature (default: 0.7)
  --top-p F                    Top-p sampling (default: 1.0)
  --seed N                     Random seed

Thinking:
  --think                      Normal thinking (default)
  --think-max                  Maximum thinking
  --no-think                   Disable thinking

Other:
  -q, --quiet                  Suppress diagnostic output
  --debug-tokens               Print token IDs alongside decoded text + top-5 logprobs
  --no-batched                 Disable batched parallel prefill (use sequential)
  -h, --help                   Show this help"
    );
}
