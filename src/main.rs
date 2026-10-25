use anyhow::{bail, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures_util::StreamExt;
use qwen3_asr::{AsrInference, StreamingOptions};
use serde_json::{json, Value};
use std::sync::Arc;
use std::path::Path;
use tokio::sync::Mutex;
use uuid::Uuid;

type SharedEngine = Arc<Mutex<AsrInference>>;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
        .format_timestamp_secs()
        .init();

    eprintln!(
        "[startup] Loading Qwen3-ASR-0.6B; first run downloads it to models/ and can take several minutes..."
    );
    let started = std::time::Instant::now();
    let device = candle_core::Device::new_cuda(0)?;
    eprintln!("[startup] Selected inference device: {device:?}");

    // The Hub API uses reqwest's blocking client, so load before starting Tokio.
    let engine = AsrInference::from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        Path::new("models/"),
        device,
    )?;
    eprintln!(
        "[startup] Model ready in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve(engine))
}

async fn serve(engine: AsrInference) -> Result<()> {
    let engine = Arc::new(Mutex::new(engine));

    let app = Router::new()
        .route("/v1/realtime", get(ws_handler))
        .route("/v1/health", get(|| async {
            axum::Json(json!({"status": "ok"}))
        }))
        .with_state(engine);

    eprintln!("[startup] Binding HTTP/WebSocket server to 0.0.0.0:9000...");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:9000").await?;

    eprintln!(
        "[ready] WebSocket: ws://localhost:9000/v1/realtime (subprotocol: realtime)"
    );
    eprintln!("[ready] Health check: http://localhost:9000/v1/health");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(engine): State<SharedEngine>,
) -> impl IntoResponse {
    ws.protocols(["realtime"])
        .on_upgrade(move |socket| handle_socket(socket, engine))
}

async fn send_event(
    socket: &mut WebSocket,
    mut event: Value,
) -> Result<()> {
    event["event_id"] = json!(format!("event_{}", Uuid::new_v4()));

    socket
        .send(Message::Text(event.to_string().into()))
        .await?;

    Ok(())
}

fn pcm16_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("PCM16 audio must contain an even number of bytes");
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|b| {
            let sample = i16::from_le_bytes([b[0], b[1]]);
            sample as f32 / 32768.0
        })
        .collect())
}

async fn handle_socket(
    mut socket: WebSocket,
    engine: SharedEngine,
) {
    let mut audio = Vec::<f32>::new();
    let mut transcript = String::new();
    let mut sample_rate = 16000_u32;

    let mut stream = {
        let engine = engine.lock().await;
        engine.init_streaming(
            StreamingOptions::default()
                .with_max_new_tokens_streaming(16)
                .with_max_new_tokens_final(256),
        )
    };

    let conversation_id = format!("conv_{}", Uuid::new_v4());
    let item_id = format!("msg_{}", Uuid::new_v4());

    if send_event(
        &mut socket,
        json!({
            "type": "conversation.created",
            "conversation": {
                "id": conversation_id,
                "object": "realtime.conversation"
            }
        }),
    )
    .await
    .is_err()
    {
        return;
    }

    while let Some(Ok(message)) = socket.next().await {
        match message {
            Message::Text(text) => {
                let event: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => {
                        let _ = send_event(
                            &mut socket,
                            json!({
                                "type": "error",
                                "error": {
                                    "type": "invalid_request_error",
                                    "code": "invalid_event",
                                    "message": "Invalid JSON event",
                                    "param": null
                                }
                            }),
                        )
                        .await;
                        continue;
                    }
                };

                match event["type"].as_str().unwrap_or("") {
                    "transcription_session.update" => {
                        sample_rate = event["session"]
                            ["input_audio_params"]["sample_rate_hz"]
                            .as_u64()
                            .unwrap_or(16000) as u32;

                        if sample_rate != 16000 {
                            let _ = send_event(
                                &mut socket,
                                json!({
                                    "type": "error",
                                    "error": {
                                        "type": "invalid_request_error",
                                        "code": "unsupported_sample_rate",
                                        "message": "This example requires 16000 Hz PCM16 input",
                                        "param": "session.input_audio_params.sample_rate_hz"
                                    }
                                }),
                            )
                            .await;
                            continue;
                        }

                        let _ = send_event(
                            &mut socket,
                            json!({
                                "type": "transcription_session.updated",
                                "session": event["session"]
                            }),
                        )
                        .await;
                    }

                    "input_audio_buffer.append" => {
                        let encoded =
                            event["audio"].as_str().unwrap_or("");

                        match STANDARD.decode(encoded) {
                            Ok(bytes) => match pcm16_to_f32(&bytes) {
                                Ok(samples) => {
                                    audio.extend_from_slice(&samples);

                                    // Feed this chunk into the model.
                                    // Model inference is serialized because
                                    // this example shares one engine.
                                    let inference_started = std::time::Instant::now();
                                    let chunk_result = {
                                        let engine = engine.lock().await;
                                        engine.feed_audio(&mut stream, &samples)
                                    };
                                    if chunk_result.is_ok() {
                                        eprintln!(
                                            "[perf] audio chunk inference: {:.2}s",
                                            inference_started.elapsed().as_secs_f64()
                                        );
                                    }

                                    match chunk_result {
                                        Ok(Some(result)) => {
                                            let new_text = result.text;

                                            // Qwen may revise previous words.
                                            // Emit only a suffix when possible.
                                            let delta = if new_text
                                                .starts_with(&transcript)
                                            {
                                                new_text
                                                    [transcript.len()..]
                                                    .to_string()
                                            } else {
                                                String::new()
                                            };

                                            transcript = new_text;

                                            if !delta.is_empty() {
                                                let _ = send_event(
                                                    &mut socket,
                                                    json!({
                                                        "type": "conversation.item.input_audio_transcription.delta",
                                                        "item_id": item_id,
                                                        "content_index": 0,
                                                        "delta": delta
                                                    }),
                                                )
                                                .await;
                                            }
                                        }
                                        Ok(None) => {}
                                        Err(err) => {
                                            eprintln!("ASR error: {err}");
                                        }
                                    }
                                }
                                Err(err) => {
                                    eprintln!("Audio decoding error: {err}");
                                }
                            },
                            Err(err) => {
                                eprintln!("Base64 decoding error: {err}");
                            }
                        }
                    }

                    "input_audio_buffer.commit"
                    | "input_audio_buffer.done" => {
                        let inference_started = std::time::Instant::now();
                        let final_text = {
                            let engine = engine.lock().await;
                            engine.finish_streaming(&mut stream)
                        };
                        eprintln!(
                            "[perf] final inference: {:.2}s",
                            inference_started.elapsed().as_secs_f64()
                        );

                        match final_text {
                            Ok(result) => {
                                transcript = result.text;

                                let _ = send_event(
                                    &mut socket,
                                    json!({
                                        "type": "conversation.item.input_audio_transcription.completed",
                                        "item_id": item_id,
                                        "content_index": 0,
                                        "transcript": transcript,
                                        "is_last_result": true,
                                        "usage": {
                                            "type": "duration",
                                            "seconds": audio.len() as f64
                                                / sample_rate as f64
                                        }
                                    }),
                                )
                                .await;
                            }
                            Err(err) => {
                                eprintln!("Final ASR error: {err}");
                            }
                        }

                        audio.clear();
                        transcript.clear();

                        // Start a fresh utterance without reloading weights.
                        let engine = engine.lock().await;
                        stream = engine.init_streaming(
                            StreamingOptions::default()
                                .with_max_new_tokens_streaming(16)
                                .with_max_new_tokens_final(256),
                        );
                    }

                    "input_audio_buffer.clear" => {
                        audio.clear();
                        transcript.clear();

                        let engine = engine.lock().await;
                        stream = engine.init_streaming(
                            StreamingOptions::default()
                                .with_max_new_tokens_streaming(16)
                                .with_max_new_tokens_final(256),
                        );

                        let _ = send_event(
                            &mut socket,
                            json!({
                                "type": "input_audio_buffer.cleared"
                            }),
                        )
                        .await;
                    }

                    _ => {
                        // Unsupported event types can be rejected here.
                    }
                }
            }

            Message::Close(_) => break,
            Message::Ping(data) => {
                if socket.send(Message::Pong(data)).await.is_err() {
                    break;
                }
            }
            _ => {}
        }
    }
}