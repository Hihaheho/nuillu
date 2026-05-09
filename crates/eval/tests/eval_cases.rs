use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    ArtifactTextField, CaseArtifact, Check, EvalCase, RubricJudge, RubricJudgeError,
    RubricJudgeInput, RubricJudgeRequest, RubricJudgeVerdict, RubricJudgeVerdictCriterion,
    evaluate_case, normalize_text_block, parse_case_file, parse_full_agent_case_file,
    parse_module_case_file, render_judge_input,
};

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

struct ModuleScopedJudge {
    called: Cell<bool>,
    rendered_inputs: RefCell<Vec<String>>,
}

impl ModuleScopedJudge {
    fn new() -> Self {
        Self {
            called: Cell::new(false),
            rendered_inputs: RefCell::new(Vec::new()),
        }
    }
}

#[async_trait(?Send)]
impl RubricJudge for ModuleScopedJudge {
    async fn judge(
        &self,
        trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError> {
        assert_eq!(
            request.context.as_deref(),
            Some(
                "Module-scoped full-agent check for 'query-vector'. Judge only the selected evidence for this module."
            )
        );
        assert_eq!(
            request.judge_inputs,
            vec![RubricJudgeInput::Output, RubricJudgeInput::Memos]
        );
        assert!(request.artifact.output.contains("first query memo"));
        assert!(request.artifact.output.contains("second query memo"));
        assert!(!request.artifact.output.contains("sensory memo"));
        let rendered = render_judge_input(trace, &request);
        assert!(rendered.contains("agent.memo_logs"));
        assert!(rendered.contains("first query memo"));
        assert!(rendered.contains("second query memo"));
        assert!(!rendered.contains("sensory memo"));
        self.rendered_inputs.borrow_mut().push(rendered);
        self.called.set(true);
        Ok(RubricJudgeVerdict {
            passed: false,
            score: 0.2,
            summary: "module diagnostic failed without failing the case".to_string(),
            criteria: Vec::new(),
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
fn rejects_duplicate_modules() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-vector");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("duplicate-modules.eure");
    std::fs::write(
        &path,
        r#"
id = "duplicate-modules"
modules = ["query-vector", "query-vector"]
prompt = "Find memory."
"#,
    )
    .unwrap();

    let err = parse_module_case_file(&path).unwrap_err();
    assert!(err.to_string().contains("duplicate module"), "{err}");
}

#[test]
fn rejects_module_case_modules_missing_target() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-vector");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("missing-target.eure");
    std::fs::write(
        &path,
        r#"
id = "missing-target"
modules = ["attention-schema"]
prompt = "Find memory."
"#,
    )
    .unwrap();

    let err = parse_case_file(&path).unwrap_err();
    assert!(
        err.to_string().contains("must include target module"),
        "{err}"
    );
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

#[test]
fn rejects_full_agent_modules_checks_for_unregistered_module() {
    let dir = tempfile::tempdir().unwrap();
    let full_agent_dir = dir.path().join("full-agent");
    std::fs::create_dir_all(&full_agent_dir).unwrap();
    let path = full_agent_dir.join("bad-module-check.eure");
    std::fs::write(
        &path,
        r#"
id = "bad-module-check"
modules = ["sensory"]

@ inputs[] {
  $variant: heard
  content = "Find the memory."
}

@ modules-checks[] {
  module = "query-vector"

  @ rubrics[] {
    rubric = "Judge query-vector output."
  }
}
"#,
    )
    .unwrap();

    let err = parse_full_agent_case_file(&path).unwrap_err();
    assert!(
        err.to_string()
            .contains("must be included in full-agent modules"),
        "{err}"
    );
}

#[tokio::test]
async fn evaluates_full_agent_modules_checks_from_scoped_memo_logs_without_affecting_score() {
    let dir = tempfile::tempdir().unwrap();
    let full_agent_dir = dir.path().join("eval-cases/full-agent");
    std::fs::create_dir_all(&full_agent_dir).unwrap();
    let path = full_agent_dir.join("module-checks-eval.eure");
    std::fs::write(
        &path,
        r#"
id = "module-checks-eval"
modules = ["sensory", "query-vector"]

@ inputs[] {
  $variant: heard
  content = "Find the memory."
}

@ modules-checks[] {
  module = "query-vector"

  @ rubrics[] {
    name = "query-history"
    pass-score = 0.85
    judge-inputs = ["output", "memos"]
    rubric = "Judge the query-vector memo history."
  }
}
"#,
    )
    .unwrap();
    let case = EvalCase::FullAgent(parse_full_agent_case_file(&path).unwrap());
    let artifact = CaseArtifact::new("final utterance").with_observation(
        "agent",
        serde_json::json!({
            "memos": {
                "query-vector": [
                    { "replica": 0, "memo": "second query memo" }
                ],
                "sensory": [
                    { "replica": 0, "memo": "sensory memo" }
                ]
            },
            "memo_logs": {
                "query-vector": [
                    {
                        "replica": 0,
                        "index": 0,
                        "written_at": "2026-05-08T00:00:00Z",
                        "content": "first query memo"
                    },
                    {
                        "replica": 0,
                        "index": 1,
                        "written_at": "2026-05-08T00:00:01Z",
                        "content": "second query memo"
                    }
                ],
                "sensory": [
                    {
                        "replica": 0,
                        "index": 0,
                        "written_at": "2026-05-08T00:00:00Z",
                        "content": "sensory memo"
                    }
                ]
            }
        }),
    );
    let judge = ModuleScopedJudge::new();

    let report = evaluate_case(&case, &empty_trace(), &artifact, Some(&judge)).await;

    assert!(judge.called.get());
    assert!(report.passed(), "{report:#?}");
    assert_eq!(report.score, 1.0);
    assert_eq!(report.modules_checks.len(), 1);
    assert_eq!(report.modules_checks[0].module, "query-vector");
    assert_eq!(report.modules_checks[0].rubrics.len(), 1);
    assert!(!report.modules_checks[0].rubrics[0].passed);
    assert!(!report.invalid);
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
