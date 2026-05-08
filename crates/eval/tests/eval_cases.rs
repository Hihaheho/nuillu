use std::{cell::Cell, path::Path};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    ArtifactTextField, CaseArtifact, Check, EvalCase, FullAgentInput, ModuleEvalTarget,
    ReasoningEffort, RubricJudge, RubricJudgeError, RubricJudgeInput, RubricJudgeRequest,
    RubricJudgeVerdict, RubricJudgeVerdictCriterion, discover_case_files, evaluate_case,
    normalize_text_block, parse_case_file, parse_full_agent_case_file, parse_model_set_file,
    parse_module_case_file, render_judge_input,
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

struct CaseDataJudge {
    called: Cell<bool>,
    expected_prompt: String,
    expected_context: Option<String>,
    expected_rubric: String,
    expected_criteria_names: Vec<String>,
    expected_pass_score: f64,
    expected_judge_inputs: Vec<RubricJudgeInput>,
    expected_judge_max_output_tokens: u32,
}

impl CaseDataJudge {
    fn from_case(case: &EvalCase) -> Self {
        let Some(Check::Rubric {
            rubric,
            criteria,
            pass_score,
            judge_inputs,
            ..
        }) = case
            .checks()
            .iter()
            .find(|check| matches!(check, Check::Rubric { .. }))
        else {
            panic!("expected case to define a rubric check");
        };

        Self {
            called: Cell::new(false),
            expected_prompt: normalize_text_block(&case.prompt_for_judge()),
            expected_context: case
                .context_for_judge()
                .map(|context| normalize_text_block(&context)),
            expected_rubric: normalize_text_block(&rubric.content),
            expected_criteria_names: criteria
                .iter()
                .map(|criterion| criterion.name.clone())
                .collect(),
            expected_pass_score: *pass_score,
            expected_judge_inputs: judge_inputs.clone(),
            expected_judge_max_output_tokens: case.scoring().judge_max_output_tokens,
        }
    }
}

#[async_trait(?Send)]
impl RubricJudge for CaseDataJudge {
    async fn judge(
        &self,
        _trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError> {
        assert_eq!(request.prompt, self.expected_prompt);
        assert_eq!(request.context, self.expected_context);
        assert_eq!(request.rubric, self.expected_rubric);
        assert_eq!(
            request
                .criteria
                .iter()
                .map(|criterion| criterion.name.clone())
                .collect::<Vec<_>>(),
            self.expected_criteria_names
        );
        assert_eq!(request.pass_score, self.expected_pass_score);
        assert_eq!(request.judge_inputs, self.expected_judge_inputs);
        assert_eq!(
            request.judge_max_output_tokens,
            self.expected_judge_max_output_tokens
        );
        assert!(!request.artifact.output.trim().is_empty());
        self.called.set(true);
        Ok(RubricJudgeVerdict {
            passed: true,
            score: 0.92,
            summary: "rubric request was built from the parsed case data".to_string(),
            criteria: request
                .criteria
                .iter()
                .map(|criterion| RubricJudgeVerdictCriterion {
                    name: criterion.name.clone(),
                    passed: true,
                    score: 0.92,
                    reason: "criterion came from the case rubric".to_string(),
                    evidence: None,
                })
                .collect(),
        })
    }
}

fn artifact_from_output_checks(case: &EvalCase) -> CaseArtifact {
    let mut output = String::new();
    for check in case.checks() {
        match check {
            Check::ArtifactTextContains {
                field, contains, ..
            } if field.unwrap_or(ArtifactTextField::Output) == ArtifactTextField::Output => {
                output.push_str(contains);
                output.push('\n');
            }
            Check::ArtifactTextExact { field, exact, .. }
                if field.unwrap_or(ArtifactTextField::Output) == ArtifactTextField::Output =>
            {
                output.push_str(&exact.content);
                output.push('\n');
            }
            _ => {}
        }
    }

    assert!(
        !output.trim().is_empty(),
        "expected case to define an output text check"
    );
    CaseArtifact::new(output)
}

#[test]
fn parses_checked_in_eval_cases() {
    let files = discover_case_files(&eval_root()).unwrap();

    assert_eq!(files.len(), 10, "expected checked-in eval case count");
    for path in files {
        parse_case_file(&path).unwrap_or_else(|error| {
            panic!("failed to parse {}: {error}", path.display());
        });
    }
}

#[test]
fn parses_checked_in_model_set() {
    let path = workspace_root().join("configs/modelsets/eval-ollama.eure");
    let model_set = parse_model_set_file(&path).unwrap();

    let judge = model_set.judge.unwrap();
    assert_eq!(judge.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(judge.token.as_deref(), Some("local"));
    assert_eq!(judge.model.as_deref(), Some("gpt-oss:20b"));
    assert_eq!(judge.reasoning_effort, None);
    assert_eq!(judge.use_responses_api, None);
    let cheap = model_set.cheap.unwrap();
    assert_eq!(cheap.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(cheap.token.as_deref(), Some("local"));
    assert_eq!(cheap.model.as_deref(), Some("gemma4:26b"));
    assert_eq!(cheap.reasoning_effort, Some(ReasoningEffort::Low));
    assert_eq!(cheap.use_responses_api, None);
    let default = model_set.default.unwrap();
    assert_eq!(default.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(default.token.as_deref(), Some("local"));
    assert_eq!(default.model.as_deref(), Some("gemma4:26b"));
    assert_eq!(default.reasoning_effort, Some(ReasoningEffort::Medium));
    assert_eq!(default.use_responses_api, None);
    let premium = model_set.premium.unwrap();
    assert_eq!(premium.endpoint(), Some("http://localhost:11434/v1"));
    assert_eq!(premium.token.as_deref(), Some("local"));
    assert_eq!(premium.model.as_deref(), Some("gemma4:26b"));
    assert_eq!(premium.reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(premium.use_responses_api, None);
}

#[test]
fn parses_responses_api_model_set_option() {
    let path = workspace_root().join("configs/modelsets/eval-gpt5.4.eure");
    let model_set = parse_model_set_file(&path).unwrap();

    assert_eq!(model_set.judge.unwrap().use_responses_api, Some(true));
    assert_eq!(model_set.cheap.unwrap().use_responses_api, Some(true));
    assert_eq!(model_set.default.unwrap().use_responses_api, Some(true));
    assert_eq!(model_set.premium.unwrap().use_responses_api, Some(true));
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
fn parses_full_agent_peer_boundary_case() {
    let path = eval_root().join("full-agent/dog-peer-food-boundary.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::FullAgent(case) = case else {
        panic!("expected full-agent case");
    };

    assert_eq!(
        case.id.as_deref(),
        Some("full-agent-dog-peer-food-boundary")
    );
    assert_eq!(case.inputs.len(), 2);
    assert!(matches!(case.inputs[0], FullAgentInput::Seen { .. }));
    assert!(matches!(case.inputs[1], FullAgentInput::Heard { .. }));
    assert_eq!(case.limits.max_llm_calls, Some(10));
}

#[test]
fn parses_full_agent_memory_required_case() {
    let path = eval_root().join("full-agent/own-body-simple-report.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::FullAgent(case) = case else {
        panic!("expected full-agent case");
    };

    assert_eq!(
        case.id.as_deref(),
        Some("full-agent-own-body-simple-report")
    );
    assert_eq!(case.inputs.len(), 2);
    assert_eq!(case.memories.len(), 1);
    assert_eq!(
        MemoryRank::from(case.memories[0].rank),
        MemoryRank::Permanent
    );
    assert!(!case.memories[0].content.content.trim().is_empty());
}

#[test]
fn omitted_max_llm_calls_defaults_to_ten() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-vector");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("default-budget.eure");
    std::fs::write(
        &path,
        r#"
id = "default-budget"
prompt = "Find memory."
"#,
    )
    .unwrap();

    let case = parse_module_case_file(&path).unwrap();
    assert_eq!(case.limits.max_llm_calls, Some(10));
}

#[test]
fn explicit_max_llm_calls_override_wins() {
    let path = eval_root().join("modules/query-vector/retrieve-koro-approach-rule.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { case, .. } = case else {
        panic!("expected module case");
    };

    assert_eq!(case.limits.max_llm_calls, Some(8));
}

#[test]
fn parses_query_vector_module_case_with_memory_seed() {
    let path = eval_root().join("modules/query-vector/retrieve-koro-approach-rule.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::QueryVector);
    assert!(!case.prompt.content.trim().is_empty());
    assert_eq!(case.memories.len(), 1);
    assert_eq!(
        MemoryRank::from(case.memories[0].rank),
        MemoryRank::Permanent
    );
    assert!(!case.memories[0].content.content.trim().is_empty());
    assert!(case.checks.iter().any(|check| matches!(
        check,
        Check::Rubric {
            judge_inputs,
            ..
        } if !judge_inputs.is_empty()
    )));
}

#[test]
fn parses_query_agentic_module_case() {
    let path = eval_root().join("modules/query-agentic/retrieve-torus-route-rule.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::QueryAgentic);
    assert!(!case.prompt.content.trim().is_empty());
    assert_eq!(case.files.len(), 1);
    assert!(!case.files[0].path.trim().is_empty());
    assert!(!case.files[0].content.content.trim().is_empty());
    assert_eq!(case.limits.max_llm_calls, Some(8));
}

#[test]
fn parses_attention_schema_module_case() {
    let path = eval_root().join("modules/attention-schema/current-attended-peer-signal.eure");
    let case = parse_case_file(&path).unwrap();

    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target, ModuleEvalTarget::AttentionSchema);
    assert!(!case.prompt.content.trim().is_empty());
    assert_eq!(case.attention_stream.len(), 1);
    assert_eq!(case.attention_stream[0].seconds_ago, 3);
    assert!(!case.attention_stream[0].text.content.trim().is_empty());
}

#[test]
fn rejects_negative_attention_seed_seconds_ago() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/attention-schema");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("negative-attention-seed.eure");
    std::fs::write(
        &path,
        r#"
id = "negative-attention-seed"
prompt = "What am I attending to?"

@ attention-stream[] {
  text = "Current attended item"
  seconds-ago = -1
}
"#,
    )
    .unwrap();

    let err = parse_module_case_file(&path).unwrap_err();
    assert!(err.to_string().contains("seconds-ago"), "{err}");
}

#[test]
fn rejects_empty_rubric_judge_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-vector");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("empty-judge-inputs.eure");
    std::fs::write(
        &path,
        r#"
id = "empty-judge-inputs"
prompt = "Find the seeded memory."

@ checks[] {
  $variant: rubric
  name = "bad-rubric"
  judge-inputs = []
  rubric = "Judge the output."
}
"#,
    )
    .unwrap();

    let err = parse_module_case_file(&path).unwrap_err();
    assert!(err.to_string().contains("judge-inputs"), "{err}");
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
    let path = eval_root().join("modules/query-vector/retrieve-koro-approach-rule.eure");
    let case = parse_case_file(&path).unwrap();
    let artifact = artifact_from_output_checks(&case);
    let judge = CaseDataJudge::from_case(&case);
    let check_count = case.checks().len();

    let report = evaluate_case(&case, &empty_trace(), &artifact, Some(&judge)).await;

    assert!(judge.called.get());
    assert!(report.passed(), "{report:#?}");
    assert_eq!(report.checks.len(), check_count);
    assert!(report.score > 0.9);
}

#[test]
fn render_judge_input_includes_only_selected_sections() {
    let artifact = CaseArtifact::new("retrieved file content only").with_observation(
        "agent",
        serde_json::json!({
            "memos": {
                "query-agentic": ["runtime metadata"]
            },
            "memory_metadata": {
                "mem-1": { "rank": "permanent" }
            }
        }),
    );
    let request = RubricJudgeRequest {
        prompt: "Find the seeded note".to_string(),
        context: Some("Judge content-only-output against artifact.output.".to_string()),
        rubric: "The primary output must contain only retrieved content.".to_string(),
        criteria: Vec::new(),
        pass_score: 0.85,
        judge_inputs: vec![RubricJudgeInput::Output, RubricJudgeInput::Memory],
        judge_max_output_tokens: 1200,
        artifact,
    };

    let rendered = render_judge_input(&empty_trace(), &request);

    assert!(rendered.contains("Primary artifact output:\nretrieved file content only"));
    assert!(rendered.contains("Memory JSON:"));
    assert!(rendered.contains("\"mem-1\""));
    assert!(!rendered.contains("Artifact observations JSON:"));
    assert!(!rendered.contains("Trace summary:"));
    assert!(!rendered.contains("\"query-agentic\""));
}

#[tokio::test]
async fn rubric_case_requires_judge() {
    let path = eval_root().join("modules/query-vector/retrieve-koro-approach-rule.eure");
    let case = parse_case_file(&path).unwrap();
    let artifact = artifact_from_output_checks(&case);

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
