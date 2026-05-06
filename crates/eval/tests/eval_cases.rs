use std::{cell::Cell, error::Error, path::Path};

use async_trait::async_trait;
use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    CaseArtifact, CaseDriver, EvalCase, RubricJudge, RubricJudgeError, RubricJudgeRequest,
    RubricJudgeVerdict, RubricJudgeVerdictCriterion, discover_case_files, parse_case_file,
    run_case,
};
use nuillu_types::MemoryRank;

struct EchoDriver;

#[async_trait(?Send)]
impl CaseDriver for EchoDriver {
    async fn run_case(&mut self, case: &EvalCase) -> Result<CaseArtifact, Box<dyn Error>> {
        let _span = tracing::info_span!("echo-driver", "lutum.capture" = true).entered();
        let seeded_memories = case
            .memories
            .iter()
            .map(|memory| memory.content.content.clone())
            .collect::<Vec<_>>();
        let output = if seeded_memories.is_empty() {
            format!("hello from nuillu: {}", case.prompt.content)
        } else {
            format!("Based on memory: {}", seeded_memories.join(" "))
        };

        Ok(CaseArtifact::new(output)
            .with_observation("seeded_memories", serde_json::json!(seeded_memories))
            .with_observation("memos", serde_json::json!({"query-agentic": "answered"})))
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

struct MemoryJudge;

#[async_trait(?Send)]
impl RubricJudge for MemoryJudge {
    async fn judge(
        &self,
        _trace: &TraceSnapshot,
        request: RubricJudgeRequest,
    ) -> Result<RubricJudgeVerdict, RubricJudgeError> {
        assert!(request.prompt.contains("Who are you?"));
        assert!(request.artifact.output.contains("blue frog"));
        Ok(RubricJudgeVerdict {
            passed: true,
            score: 0.95,
            summary: "answer uses the seeded permanent memory".to_string(),
            criteria: vec![RubricJudgeVerdictCriterion {
                name: "uses-seeded-memory".to_string(),
                passed: true,
                score: 0.95,
                reason: "the output includes Lutum and blue frog identity".to_string(),
                evidence: Some("I'm a Lutum, a blue frog.".to_string()),
            }],
        })
    }
}

#[test]
fn parses_checked_in_eval_cases() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases");
    let files = discover_case_files(&root).unwrap();

    assert!(!files.is_empty(), "expected checked-in eval cases");
    for path in files {
        parse_case_file(&path).unwrap_or_else(|error| {
            panic!("failed to parse {}: {error}", path.display());
        });
    }
}

#[test]
fn parses_memory_seed_setup() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases/smoke/memory-identity.eure");

    let case = parse_case_file(&path).unwrap();

    assert_eq!(case.memories.len(), 1);
    assert_eq!(case.memories[0].key.as_deref(), Some("identity"));
    assert_eq!(
        MemoryRank::from(case.memories[0].rank),
        MemoryRank::Permanent
    );
    assert_eq!(
        case.memories[0].content.content,
        "I'm a Lutum, a blue frog."
    );
}

#[tokio::test]
async fn runs_case_with_driver_and_rubric_judge() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases/smoke/greeting.eure");
    let mut driver = EchoDriver;
    let judge = ScriptJudge {
        called: Cell::new(false),
    };

    let summary = run_case(&path, &mut driver, Some(&judge)).await.unwrap();

    assert!(judge.called.get());
    assert!(summary.passed, "{:#?}", summary.report);
    assert_eq!(summary.id, "smoke-greeting");
    assert_eq!(summary.report.checks.len(), 3);
    assert!(summary.score > 0.9);
}

#[tokio::test]
async fn rubric_case_requires_judge() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases/smoke/greeting.eure");
    let mut driver = EchoDriver;

    let summary = run_case(&path, &mut driver, None).await.unwrap();

    assert!(!summary.passed);
    assert!(summary.invalid);
    assert!(
        summary
            .report
            .checks
            .iter()
            .any(|check| check.kind == "rubric" && check.errored)
    );
}

#[tokio::test]
async fn seeded_memory_is_available_to_driver() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval-cases/smoke/memory-identity.eure");
    let mut driver = EchoDriver;

    let summary = run_case(&path, &mut driver, Some(&MemoryJudge))
        .await
        .unwrap();

    assert!(summary.passed, "{:#?}", summary.report);
    assert_eq!(summary.id, "smoke-memory-identity");
}
