use std::{cell::Cell, path::Path};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    CaseArtifact, EvalCase, FullAgentInput, ModuleEvalTarget, ReasoningEffort, RubricJudge,
    RubricJudgeError, RubricJudgeRequest, RubricJudgeVerdict, RubricJudgeVerdictCriterion,
    discover_case_files, evaluate_case, parse_case_file, parse_full_agent_case_file,
    parse_model_set_file,
};
use nuillu_types::MemoryRank;

fn eval_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases")
}

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn empty_trace() -> TraceSnapshot {
    TraceSnapshot {
        roots: Vec::new(),
        root_events: Vec::new(),
    }
}

struct ScriptJudge {
    called: Cell<bool>,
}

#[async_trait(?Send)]
impl RubricJudge for ScriptJudge {
    async fn judge(
        &self,
        _trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError> {
        assert!(request.prompt.contains("Greet the user"));
        assert!(request.rubric.contains("brief greeting"));
        assert!(request.artifact.output.contains("nuillu"));
        self.called.set(true);
        Ok(RubricJudgeVerdict {
            passed: true,
            score: 0.92,
            summary: "brief greeting with correct identity".to_string(),
            criteria: vec![
                RubricJudgeVerdictCriterion {
                    name: "greeting".to_string(),
                    passed: true,
                    score: 0.9,
                    reason: "starts with a greeting".to_string(),
                    evidence: Some("hello".to_string()),
                },
                RubricJudgeVerdictCriterion {
                    name: "identity".to_string(),
                    passed: true,
                    score: 1.0,
                    reason: "names nuillu".to_string(),
                    evidence: Some("hello from nuillu".to_string()),
                },
            ],
        })
    }
}

#[test]
fn parses_checked_in_eval_cases() {
    let files = discover_case_files(&eval_root()).unwrap();

    assert!(!files.is_empty(), "expected checked-in eval cases");
    for path in files {
        parse_case_file(&path).unwrap_or_else(|error| {
            panic!("failed to parse {}: {error}", path.display());
        });
    }
}

#[test]
fn parses_checked_in_model_set() {
    let path = workspace_root().join("configs/modelsets/eval.eure");
    let model_set = parse_model_set_file(&path).unwrap();

    let judge = model_set.judge.unwrap();
    assert_eq!(judge.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(judge.token.as_deref(), Some("local"));
    assert_eq!(judge.model.as_deref(), Some("gpt-oss:20b"));
    assert_eq!(judge.reasoning_effort, None);
    let cheap = model_set.cheap.unwrap();
    assert_eq!(cheap.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(cheap.token.as_deref(), Some("local"));
    assert_eq!(cheap.model.as_deref(), Some("gemma4:e2b"));
    assert_eq!(cheap.reasoning_effort, None);
    let default = model_set.default.unwrap();
    assert_eq!(default.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(default.token.as_deref(), Some("local"));
    assert_eq!(default.model.as_deref(), Some("gemma4:e4b"));
    assert_eq!(default.reasoning_effort, None);
    let premium = model_set.premium.unwrap();
    assert_eq!(premium.endpoint(), Some("https://openrouter.ai/api/v1"));
    assert_eq!(premium.token_env.as_deref(), Some("OPENROUTER_API_KEY"));
    assert_eq!(premium.model.as_deref(), Some("google/gemma-4-26b-a4b-it"));
    assert_eq!(premium.reasoning_effort, Some(ReasoningEffort::Medium));
}

#[test]
fn rejects_global_model_set_backend_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("global-backend.eure");
    std::fs::write(
        &path,
        r#"
name = "bad-global"
endpoint = "http://localhost:11434/v1"

judge {
  model = "judge"
}
"#,
    )
    .unwrap();

    let err = parse_model_set_file(&path).unwrap_err();
    assert!(err.to_string().contains("endpoint"), "{err}");
}

#[test]
fn parses_full_agent_multimodal_case() {
    let path = eval_root().join("full-agent/multimodal-greeting.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::FullAgent(case) = case else {
        panic!("expected full-agent case");
    };

    assert_eq!(case.id.as_deref(), Some("full-agent-multimodal-greeting"));
    assert_eq!(case.inputs.len(), 2);
    assert!(matches!(case.inputs[0], FullAgentInput::Heard { .. }));
    assert!(matches!(case.inputs[1], FullAgentInput::Seen { .. }));
    assert_eq!(case.limits.max_llm_calls, Some(64));
}

#[test]
fn parses_query_vector_module_case_with_memory_seed() {
    let path = eval_root().join("modules/query-vector/memory-identity.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::QueryVector);
    assert_eq!(case.prompt.content, "Who are you?");
    assert_eq!(case.memories.len(), 1);
    assert_eq!(
        MemoryRank::from(case.memories[0].rank),
        MemoryRank::Permanent
    );
    assert_eq!(
        case.memories[0].content.content,
        "I'm a Lutum, a blue frog."
    );
}

#[test]
fn parses_query_agentic_module_case() {
    let path = eval_root().join("modules/query-agentic/greeting.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::QueryAgentic);
    assert!(case.prompt.content.contains("Greet the user"));
    assert_eq!(case.limits.max_llm_calls, Some(8));
}

#[test]
fn parses_attention_schema_module_case() {
    let path = eval_root().join("modules/attention-schema/self-report.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::AttentionSchema);
    assert!(case.prompt.content.contains("aware"));
}

#[test]
fn rejects_internal_message_variants_in_full_agent_case() {
    let dir = tempfile::tempdir().unwrap();
    let full_agent_dir = dir.path().join("full-agent");
    std::fs::create_dir_all(&full_agent_dir).unwrap();

    for variant in ["query-request", "self-model-request"] {
        let path = full_agent_dir.join(format!("{variant}.eure"));
        std::fs::write(
            &path,
            format!(
                r#"
id = "bad-full-agent-internal-message"

@ inputs[] {{
  $variant: {variant}
  question = "Who are you?"
}}
"#
            ),
        )
        .unwrap();

        let err = parse_full_agent_case_file(&path).unwrap_err();
        assert!(err.to_string().contains("parse"), "{err}");
    }
}

#[tokio::test]
async fn evaluates_module_case_with_rubric_judge() {
    let path = eval_root().join("modules/query-agentic/greeting.eure");
    let case = parse_case_file(&path).unwrap();
    let artifact = CaseArtifact::new("hello from nuillu");
    let judge = ScriptJudge {
        called: Cell::new(false),
    };

    let report = evaluate_case(&case, &empty_trace(), &artifact, Some(&judge)).await;

    assert!(judge.called.get());
    assert!(report.passed(), "{report:#?}");
    assert_eq!(report.checks.len(), 2);
    assert!(report.score > 0.9);
}

#[tokio::test]
async fn rubric_case_requires_judge() {
    let path = eval_root().join("modules/query-agentic/greeting.eure");
    let case = parse_case_file(&path).unwrap();
    let artifact = CaseArtifact::new("hello from nuillu");

    let report = evaluate_case(&case, &empty_trace(), &artifact, None).await;

    assert!(!report.passed());
    assert!(report.invalid);
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.kind == "rubric" && check.errored)
    );
}
