use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    CaseArtifact, EvalCase, RubricJudge, RubricJudgeError, RubricJudgeInput, RubricJudgeRequest,
    RubricJudgeVerdict, evaluate_case, parse_case_file, parse_full_agent_case_file,
    parse_module_case_file, render_judge_input,
};

fn empty_trace() -> TraceSnapshot {
    TraceSnapshot {
        roots: Vec::new(),
        root_events: Vec::new(),
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
fn parses_cognition_log_and_memo_seeds() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/attention-schema");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("cognition-and-memo-seeds.eure");
    std::fs::write(
        &path,
        r#"
id = "cognition-and-memo-seeds"
prompt = "What am I attending to?"

@ cognition-log[] {
  text = "A promoted cognitive item"
  seconds-ago = 4
}

@ memos[] {
  module = "sensory"
  replica = 1
  content = "A memo-surface item"
  seconds-ago = 2
}
"#,
    )
    .unwrap();

    let case = parse_module_case_file(&path).unwrap();

    assert_eq!(case.cognition_log.len(), 1);
    assert_eq!(
        case.cognition_log[0].text.content,
        "A promoted cognitive item"
    );
    assert_eq!(case.cognition_log[0].seconds_ago, 4);
    assert_eq!(case.memos.len(), 1);
    assert_eq!(case.memos[0].module, "sensory");
    assert_eq!(case.memos[0].replica, 1);
    assert_eq!(case.memos[0].content.content, "A memo-surface item");
    assert_eq!(case.memos[0].seconds_ago, 2);
}

#[test]
fn rejects_negative_cognition_log_seed_seconds_ago() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/attention-schema");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("negative-cognition-seed.eure");
    std::fs::write(
        &path,
        r#"
id = "negative-cognition-seed"
prompt = "What am I attending to?"

@ cognition-log[] {
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
            "memo_logs": {
                "query-agentic": [{
                    "replica": 0,
                    "index": 0,
                    "written_at": "2026-05-08T00:00:00Z",
                    "content": "runtime metadata"
                }]
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
