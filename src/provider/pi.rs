use std::process::Stdio;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::ProviderError;
use crate::ai_prompt::AIPrompt;

pub(super) async fn complete(prompt: AIPrompt, model: &str) -> Result<String, ProviderError> {
    run(Command::new("pi"), prompt, model).await
}

async fn run(mut command: Command, prompt: AIPrompt, model: &str) -> Result<String, ProviderError> {
    command.args([
        "--print",
        "--no-session",
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--system-prompt",
        &prompt.system_prompt,
        "--model",
        model,
    ]);

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(ProviderError::PiIo)?;
    let mut stdin = child.stdin.take().expect("stdin is piped");

    // Drain output while writing: diffs can exceed both argument and pipe limits.
    let write_prompt = async move {
        stdin.write_all(prompt.user_prompt.as_bytes()).await?;
        stdin.shutdown().await
    };
    let (written, output) = tokio::join!(write_prompt, child.wait_with_output());
    let output = output.map_err(ProviderError::PiIo)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ProviderError::PiFailed(format!(
            "{}: {}",
            output.status,
            stderr.trim()
        )));
    }
    written.map_err(ProviderError::PiIo)?;

    let response = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if response.is_empty() {
        return Err(ProviderError::NoCompletionChoice);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_MODEL: &str = "openai-codex/gpt-6.1-terra";

    fn prompt() -> AIPrompt {
        AIPrompt {
            system_prompt: "Output only the answer.".to_string(),
            user_prompt: "A diff with 'quotes', $variables, and\nnewlines".to_string(),
        }
    }

    #[tokio::test]
    async fn missing_executable_is_actionable() {
        let dir = tempfile::tempdir().unwrap();
        let error = run(
            Command::new(dir.path().join("missing-pi")),
            prompt(),
            DEFAULT_MODEL,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, ProviderError::PiIo(_)));
        assert!(error.to_string().contains("installed and on PATH"));
    }

    #[test]
    fn provider_uses_codex_terra_default_or_model_override() {
        use crate::config::cli::ProviderType;
        use crate::provider::LumenProvider;

        let provider = LumenProvider::new(ProviderType::Pi, None, None).unwrap();
        assert_eq!(
            provider.to_string(),
            "pi (CLI) (openai-codex/gpt-6.1-terra)"
        );
        match &provider.backend {
            super::super::ProviderBackend::Pi { model } => assert_eq!(model, DEFAULT_MODEL),
            _ => panic!("expected pi backend"),
        }
        let provider = LumenProvider::new(
            ProviderType::Pi,
            None,
            Some("openai/gpt-5-mini".to_string()),
        )
        .unwrap();
        assert_eq!(provider.to_string(), "pi (CLI) (openai/gpt-5-mini)");
    }

    #[cfg(unix)]
    mod unix {
        use super::*;

        fn shell(script: &str) -> Command {
            let mut command = Command::new("sh");
            command.args(["-c", script, "--"]);
            command
        }

        #[tokio::test]
        async fn passes_isolation_flags_and_prompt_without_a_shell() {
            let response = run(shell("printf '%s\\n' \"$@\"; cat"), prompt(), DEFAULT_MODEL)
                .await
                .unwrap();
            assert_eq!(
                response,
                format!(
                    "--print\n--no-session\n--no-tools\n--no-extensions\n--no-skills\n\
                     --no-prompt-templates\n--no-context-files\n--system-prompt\n{}\n--model\n{}\n{}",
                    prompt().system_prompt,
                    DEFAULT_MODEL,
                    prompt().user_prompt
                )
            );
        }

        #[tokio::test]
        async fn forwards_model_as_one_argument() {
            let response = run(
                shell("cat >/dev/null; printf '%s\\n' \"$@\""),
                prompt(),
                "openai/gpt-5-mini:low",
            )
            .await
            .unwrap();
            assert!(response.ends_with("--model\nopenai/gpt-5-mini:low"));
        }

        #[tokio::test]
        async fn handles_large_prompts_and_output_without_deadlocking() {
            let prompt = AIPrompt {
                user_prompt: "diff\n".repeat(100_000),
                ..prompt()
            };
            let response = run(
                shell("head -c 100000 /dev/zero | tr '\\0' x; cat >/dev/null; echo warning >&2"),
                prompt,
                DEFAULT_MODEL,
            )
            .await
            .unwrap();
            assert_eq!(response, "x".repeat(100_000));
        }

        #[tokio::test]
        async fn reports_nonzero_exit_and_stderr_instead_of_partial_output() {
            let error = run(
                shell("echo partial; echo 'Please log in' >&2; exit 7"),
                prompt(),
                DEFAULT_MODEL,
            )
            .await
            .unwrap_err();
            assert!(matches!(error, ProviderError::PiFailed(_)));
            assert!(error.to_string().contains("Please log in"));
            assert!(error.to_string().contains('7'));
        }

        #[tokio::test]
        async fn rejects_empty_output() {
            let error = run(
                shell("cat >/dev/null; printf ' \\n'"),
                prompt(),
                DEFAULT_MODEL,
            )
            .await
            .unwrap_err();
            assert!(matches!(error, ProviderError::NoCompletionChoice));
        }
    }
}
