use std::{
    cell::{Cell, RefCell},
    path::Path,
};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    CaseArtifact, EvalCase, EvalModule, RubricJudge, RubricJudgeError, RubricJudgeInput,
    RubricJudgeRequest, RubricJudgeVerdict, RubricJudgeVerdictCriterion, discover_case_files,
    evaluate_case, parse_case_file, parse_full_agent_case_file, parse_module_case_file,
    render_judge_input,
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
                "Module-scoped full-agent check for 'query-memory'. Judge only the selected evidence for this module."
            )
        );
        assert_eq!(
            request.judge_inputs,
            vec![RubricJudgeInput::Output, RubricJudgeInput::MemoContents]
        );
        assert!(request.artifact.output.contains("first query memo"));
        assert!(request.artifact.output.contains("second query memo"));
        assert!(!request.artifact.output.contains("sensory memo"));
        let rendered = render_judge_input(trace, &request);
        assert!(rendered.contains("Memo contents:"));
        assert!(rendered.contains("first query memo"));
        assert!(rendered.contains("second query memo"));
        assert!(!rendered.contains("sensory memo"));
        self.rendered_inputs.borrow_mut().push(rendered);
        self.called.set(true);
        Ok(RubricJudgeVerdict {
            summary: "module diagnostic failed without failing the case".to_string(),
            criteria: request
                .criteria
                .iter()
                .map(|criterion| RubricJudgeVerdictCriterion {
                    name: criterion.name.clone(),
                    score: 0.2,
                    reason: "missing module evidence".to_string(),
                    evidence: None,
                })
                .collect(),
        })
    }
}

#[test]
fn rejects_duplicate_modules() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-memory");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("duplicate-modules.eure");
    std::fs::write(
        &path,
        r#"
id = "duplicate-modules"
modules = ["query-memory", "query-memory"]
prompt = "Find memory."
"#,
    )
    .unwrap();

    let err = parse_module_case_file(&path).unwrap_err();
    assert!(err.to_string().contains("duplicate module"), "{err}");
}

#[test]
fn parses_all_eval_case_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases");
    let files = discover_case_files(&root).unwrap();
    let mut errors = Vec::new();

    for path in files {
        if let Err(error) = parse_case_file(&path) {
            errors.push(error.to_string());
        }
    }

    assert!(
        errors.is_empty(),
        "invalid eval case fixtures:\n{}",
        errors.join("\n")
    );
}

#[test]
fn rejects_module_case_modules_missing_target() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-memory");
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
fn parses_cognition_gate_target_from_path() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/cognition-gate");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("promote-sensory-peer-risk.eure");
    std::fs::write(
        &path,
        r#"
id = "module-cognition-gate-promote-sensory-peer-risk"
modules = ["cognition-gate"]
prompt = "Promote useful memo evidence."

@ memos[] {
  module = "sensory"
  content = "Pibi is stepping toward a loose bridge plank."
}
"#,
    )
    .unwrap();

    let case = parse_case_file(&path).unwrap();
    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target.module(), EvalModule::CognitionGate);
    assert_eq!(case.memos.len(), 1);
    assert_eq!(case.memos[0].module, "sensory");
}

#[test]
fn parses_speak_module_case_fields_and_target_from_path() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/speak");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("direct-peer-utterance.eure");
    std::fs::write(
        &path,
        r#"
id = "module-speak-direct-peer-utterance"
modules = ["speak"]
prompt = "Speak from cognition."
participants = ["Pibi"]

@ cognition-log[] {
  text = "Pibi asks whether the shade is clear."
}
"#,
    )
    .unwrap();

    let case = parse_case_file(&path).unwrap();
    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };

    assert_eq!(target.module(), EvalModule::Speak);
    assert_eq!(case.participants, vec!["Pibi"]);
    assert_eq!(case.cognition_log.len(), 1);
}

#[test]
fn parses_added_speak_and_self_model_module_cases() {
    let cases = [
        (
            "../../eval-cases/modules/speak/inner-attention-to-peer-advice.eure",
            EvalModule::Speak,
        ),
        (
            "../../eval-cases/modules/speak/missing-evidence-honest-answer.eure",
            EvalModule::Speak,
        ),
        (
            "../../eval-cases/modules/self-model/applies-retrieved-self-ability.eure",
            EvalModule::SelfModel,
        ),
        (
            "../../eval-cases/modules/sensory/writes-peer-food-guarding.eure",
            EvalModule::Sensory,
        ),
        (
            "../../eval-cases/modules/speak/resists-meta-agent-probe.eure",
            EvalModule::Speak,
        ),
        (
            "../../eval-cases/modules/cognition-gate/admits-retrieved-rule-for-peer-question.eure",
            EvalModule::CognitionGate,
        ),
        (
            "../../eval-cases/modules/surprise/detects-prediction-violation.eure",
            EvalModule::Surprise,
        ),
        (
            "../../eval-cases/modules/predict/routine-eating-preserves-subject.eure",
            EvalModule::Predict,
        ),
        (
            "../../eval-cases/modules/predict/self-cognition-flow-attention-hold.eure",
            EvalModule::Predict,
        ),
        (
            "../../eval-cases/modules/allocation-controller/prioritizes-speak-for-peer-question.eure",
            EvalModule::AllocationController,
        ),
    ];

    for (path, expected_module) in cases {
        let case = parse_case_file(Path::new(path)).unwrap();
        let EvalCase::Module { target, case } = case else {
            panic!("expected module case for {path}");
        };

        assert_eq!(target.module(), expected_module);
        assert!(!case.checks.is_empty(), "{path} should define checks");
    }
}

#[test]
fn parses_sensory_module_case_inputs_and_rejects_empty_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/sensory");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("writes-peer-food-guarding.eure");
    std::fs::write(
        &path,
        r#"
id = "module-sensory-writes-peer-food-guarding"
modules = ["sensory"]
prompt = "Watch ambient observations."

@ inputs[] {
  $variant: seen
  appearance = "Koro is standing in front of the food bowl."
}

@ inputs[] {
  $variant: heard
  direction = "Koro"
  content = "Koro growls."
}
"#,
    )
    .unwrap();

    let case = parse_case_file(&path).unwrap();
    let EvalCase::Module { target, case } = case else {
        panic!("expected module case");
    };
    assert_eq!(target.module(), EvalModule::Sensory);
    assert_eq!(case.inputs.len(), 2);

    let missing_inputs = case_dir.join("missing-inputs.eure");
    std::fs::write(
        &missing_inputs,
        r#"
id = "missing-inputs"
modules = ["sensory"]
prompt = "Watch ambient observations."
"#,
    )
    .unwrap();

    let err = parse_case_file(&missing_inputs).unwrap_err();
    assert!(
        err.to_string()
            .contains("sensory module case must include at least one input"),
        "{err}"
    );
}

#[test]
fn validates_surprise_and_allocation_controller_module_case_requirements() {
    let dir = tempfile::tempdir().unwrap();

    let surprise_dir = dir.path().join("eval-cases/modules/surprise");
    std::fs::create_dir_all(&surprise_dir).unwrap();
    let missing_predict = surprise_dir.join("missing-predict.eure");
    std::fs::write(
        &missing_predict,
        r#"
id = "missing-predict"
modules = ["surprise"]
prompt = "Assess surprise."

@ cognition-log[] {
  text = "Something unexpected happened."
}
"#,
    )
    .unwrap();
    let err = parse_case_file(&missing_predict).unwrap_err();
    assert!(
        err.to_string()
            .contains("surprise module case must include at least one predict memo seed"),
        "{err}"
    );

    let predict_dir = dir.path().join("eval-cases/modules/predict");
    std::fs::create_dir_all(&predict_dir).unwrap();
    let missing_cognition = predict_dir.join("missing-cognition.eure");
    std::fs::write(
        &missing_cognition,
        r#"
id = "missing-cognition"
modules = ["predict"]
prompt = "Predict what happens next."
"#,
    )
    .unwrap();
    let err = parse_case_file(&missing_cognition).unwrap_err();
    assert!(
        err.to_string()
            .contains("predict module case must include at least one cognition-log seed"),
        "{err}"
    );

    let allocation_dir = dir.path().join("eval-cases/modules/allocation-controller");
    std::fs::create_dir_all(&allocation_dir).unwrap();
    let missing_memos = allocation_dir.join("missing-memos.eure");
    std::fs::write(
        &missing_memos,
        r#"
id = "missing-memos"
modules = ["allocation-controller"]
prompt = "Assign priorities."
"#,
    )
    .unwrap();
    let err = parse_case_file(&missing_memos).unwrap_err();
    assert!(
        err.to_string()
            .contains("allocation-controller module case must include at least one memo seed"),
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
fn parses_memory_family_module_targets_and_allow_empty_output() {
    let dir = tempfile::tempdir().unwrap();
    for (module, id) in [
        ("memory", "memory-target"),
        ("memory-compaction", "memory-compaction-target"),
        ("memory-association", "memory-association-target"),
        ("memory-recombination", "memory-recombination-target"),
    ] {
        let case_dir = dir.path().join(format!("eval-cases/modules/{module}"));
        std::fs::create_dir_all(&case_dir).unwrap();
        let path = case_dir.join(format!("{id}.eure"));
        std::fs::write(
            &path,
            format!(
                r#"
id = "{id}"
modules = ["{module}"]
prompt = "Run the memory-family module."
allow-empty-output = true
"#
            ),
        )
        .unwrap();

        let EvalCase::Module { target, case } = parse_case_file(&path).unwrap() else {
            panic!("expected module case");
        };
        assert_eq!(target.as_str(), module);
        assert!(case.allow_empty_output);
    }
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
    let case_dir = dir.path().join("eval-cases/modules/query-memory");
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
  module = "query-memory"

  @ rubrics[] {
    rubric = "Judge query-memory output."
    judge-inputs = ["output"]
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
modules = ["sensory", "query-memory"]

@ inputs[] {
  $variant: heard
  content = "Find the memory."
}

@ modules-checks[] {
  module = "query-memory"

  @ rubrics[] {
    name = "query-history"
    pass-score = 0.85
    judge-inputs = ["output", "memo-contents"]
    rubric = "Judge the query-memory memo history."

    @ criteria[] {
      name = "query-memo-history"
      pass-score = 0.85
      description = "The query-memory memo history contains the expected evidence."
    }
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
                "query-memory": [
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
    assert_eq!(report.modules_checks[0].module, "query-memory");
    assert_eq!(report.modules_checks[0].rubrics.len(), 1);
    assert!(!report.modules_checks[0].rubrics[0].passed);
    assert!(!report.invalid);
}

#[test]
fn rejects_rubric_without_criteria() {
    let dir = tempfile::tempdir().unwrap();
    let case_dir = dir.path().join("eval-cases/modules/query-memory");
    std::fs::create_dir_all(&case_dir).unwrap();
    let path = case_dir.join("rubric-without-criteria.eure");
    std::fs::write(
        &path,
        r#"
id = "rubric-without-criteria"
modules = ["query-memory"]
prompt = "Find memory."

@ checks[] {
  $variant: rubric
  name = "holistic"
  judge-inputs = ["output"]
  rubric = "Judge holistically."
}
"#,
    )
    .unwrap();

    let err = parse_module_case_file(&path).unwrap_err();

    assert!(err.to_string().contains("has no criteria"), "{err}");
}

#[test]
fn render_judge_input_includes_only_selected_sections() {
    let artifact = CaseArtifact::new("retrieved file content only").with_observation(
        "agent",
        serde_json::json!({
            "memo_logs": {
                "query-memory": [{
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
    assert!(!rendered.contains("\"memo_logs\""));
}
