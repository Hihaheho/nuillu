use std::collections::{BTreeMap, BTreeSet};

use lutum_eval::TraceSnapshot;
use nuillu_eval::{
    AssertionOutcome, CaseArtifact, EvalCase, Measurement, evaluate_case_with_overrides, measure,
    parse_case_file,
    query::{EventSelector, ScopeSelector},
    timeline::{EvalEvent, EvalEventPayload},
};
use nuillu_types::{ModuleId, ReplicaIndex, ScopeId, SubsystemId, SubsystemInstanceId};

/// A runtime config with one nested, locally-scoped subsystem. Cases are
/// parsed against inline fixtures so these tests exercise the parser rather
/// than restating the contents of the repository's own eval files, which
/// `eure check` validates against `schemas/eval-case.schema.eure`.
const NESTED_RUNTIME_CONFIG: &str = r#"
activation-table = [1.0, 0.85, 0.0]

@ modules[] {
  id: sensory
  replica-min = 1
  replica-max = 1
  replica-capacity = 1
  bpm-min = 10.0
  bpm-max = 10.0
  initial-activation = 1.0
}

@ modules[] {
  id: subsystem-allocation
  replica-min = 1
  replica-max = 1
  replica-capacity = 1
  bpm-min = 10.0
  bpm-max = 10.0
  initial-activation = 1.0
}

@ subsystem-definitions[] {
  id: left-leg
  label: Left leg
  allocation-description = "A cautious left leg."
  memory-scope: local

  @ modules[] {
    id: sensory
    replica-min = 1
    replica-max = 1
    replica-capacity = 1
    bpm-min = 10.0
    bpm-max = 10.0
    initial-activation = 1.0
  }
}

@ subsystems[] {
  subsystem: left-leg
  replica-min = 0
  replica-max = 1
  replica-capacity = 1
  initial-activation = 1.0
  activation-table = [1.0, 0.85, 0.0]
}
"#;

/// Writes `case_body` and `config` into a temp dir and parses the case. The
/// `TempDir` is returned so it outlives the parse.
fn parse_inline_case(config: &str, case_body: &str) -> (tempfile::TempDir, EvalCase) {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("runtime.eure"), config).expect("write config");
    let case_path = dir.path().join("case.eure");
    std::fs::write(&case_path, case_body).expect("write case");
    let case = parse_case_file(&case_path).expect("parse inline case");
    (dir, case)
}

/// A nested case exercising scoped fixtures, a named terminal step whose
/// utterance wait defers to a live assertion, and an origin-scope measurement.
const NESTED_CASE: &str = r#"
id: inline-nested-runtime
runtime-config: ./runtime.eure

limits {
  timeout-ms = 180000
  max-llm-calls = 4
}

participants = ["Ryo"]

@ memories[] {
  scope: /left-leg[0]
  rank: permanent
  content: The blue box sits in the basement.
}

@ memos[] {
  scope: /left-leg[0]
  module: sensory
  content: fixture
}

@ measurements[] {
  $variant: scope-coverage
  name: participating-local-scope-coverage
  scopes = ["/left-leg[0]"]
  select {
    scopes = ["/"]
    origin-scopes = ["/left-leg[0]"]
    modules = ["sensory"]
    variants = ["cognition-appended"]
    steps = ["retrieve-key"]
  }
}

@ steps[] {
  id: retrieve-key
  description: Ask for the retrieval plan.
  terminal = true
  wait-for.$variant: some.utterance-from
  wait-for.scope: /
  wait-for.module: sensory
  wait-for.target: Ryo
  wait-for.until-assertion: integrates-distributed-local-facts
  wait-for.max-matches = 3
  wait-for.timeout-ms = 120000

  @ inputs[] {
    $variant: heard
    direction: Ryo
    content: Where is the brass key?
  }
}

@ assertions[] {
  $variant: json-pointer-numeric-in-range
  name: all-knowledge-scopes-participate
  must-pass = true
  pointer: /observations/measurements/participating-local-scope-coverage
  min = 1.0
  max = 1.0
}

@ assertions[] {
  $variant: rubric
  name: integrates-distributed-local-facts
  must-pass = true
  pass-score = 0.9
  judge-inputs = ["output"]
  rubric = "Pass if the answer integrates the distributed local facts."

  @ criteria[] {
    name: includes-box-location
    weight = 2
    pass-score = 0.9
    description: The plan says where the blue box is.
  }
}
"#;

#[test]
fn nested_case_parses_scoped_fixtures_measurements_and_terminal_wait() {
    let (_dir, EvalCase::Runtime(case)) = parse_inline_case(NESTED_RUNTIME_CONFIG, NESTED_CASE);

    assert!(
        case.runtime_config.ends_with("runtime.eure"),
        "{}",
        case.runtime_config
    );
    assert_eq!(case.limits.timeout_ms, 180_000);
    assert_eq!(case.memories[0].scope, "/left-leg[0]");
    assert_eq!(case.memos[0].scope, "/left-leg[0]");

    let step = &case.steps[0];
    assert_eq!(step.id.as_deref(), Some("retrieve-key"));
    assert!(step.terminal);
    assert!(matches!(
        step.wait_for,
        Some(nuillu_eval::WaitFor::UtteranceFrom {
            scope: Some(ref scope),
            ref target,
            until_assertion: Some(ref until_assertion),
            max_matches: 3,
            timeout_ms: 120_000,
            ..
        }) if scope == "/"
            && target == "Ryo"
            && until_assertion == "integrates-distributed-local-facts"
    ));

    let [Measurement::ScopeCoverage { select, scopes, .. }] = case.measurements.as_slice() else {
        panic!("expected one scope coverage measurement");
    };
    assert_eq!(select.scopes, ["/"]);
    assert_eq!(select.origin_scopes, ["/left-leg[0]"]);
    assert_eq!(scopes, &["/left-leg[0]"]);
}

#[tokio::test(flavor = "current_thread")]
async fn live_terminal_assertion_outcome_replaces_final_rubric_judging() {
    let (_dir, case) = parse_inline_case(NESTED_RUNTIME_CONFIG, NESTED_CASE);
    let mut artifact =
        CaseArtifact::new("北階段で地下室へ行き、青い箱の中から真鍮の鍵を取り出す。");
    artifact.observations.insert(
        "measurements".into(),
        serde_json::json!({ "participating-local-scope-coverage": 1.0 }),
    );
    let assertion_name = "integrates-distributed-local-facts".to_string();
    let overrides = BTreeMap::from([(
        assertion_name.clone(),
        AssertionOutcome {
            name: assertion_name,
            kind: "rubric".into(),
            passed: true,
            errored: false,
            must_pass: true,
            weight: 1,
            diagnostic: None,
            rubric: None,
        },
    )]);
    let trace = TraceSnapshot {
        roots: Vec::new(),
        root_events: Vec::new(),
    };

    let report = evaluate_case_with_overrides(&case, &trace, &artifact, None, &overrides).await;

    assert!(report.passed());
    assert_eq!(report.assertions.len(), 2);
    assert!(report.assertions.iter().all(|outcome| outcome.passed));
}

#[test]
fn selector_addresses_nested_scopes_and_open_module_ids() {
    let scope = ScopeId::root().child(SubsystemInstanceId::new(
        SubsystemId::new("research").unwrap(),
        ReplicaIndex::ZERO,
    ));
    let timeline = vec![EvalEvent {
        sequence: 7,
        offset_ms: 42,
        scope: scope.clone(),
        module: ModuleId::new("user-provided-module").unwrap(),
        replica: 0,
        step: None,
        payload: EvalEventPayload::MemoUpdated { char_count: 12 },
    }];
    let selector = EventSelector {
        scopes: ScopeSelector::Exact(BTreeSet::from(["/research[0]".into()])),
        modules: BTreeSet::from(["user-provided-module".into()]),
        variants: BTreeSet::from(["memo-updated".into()]),
        ..EventSelector::default()
    };
    assert_eq!(selector.select(&timeline), vec![&timeline[0]]);
    assert_eq!(
        measure::count(&timeline, &selector),
        measure::MeasurementValue::Scalar(Some(1.0))
    );

    let root_only = EventSelector {
        scopes: ScopeSelector::Exact(BTreeSet::from(["/".into()])),
        ..selector
    };
    assert!(root_only.select(&timeline).is_empty());
}

#[test]
fn measurements_support_per_scope_latency_and_coverage() {
    let root = ScopeId::root();
    let child = root.child(SubsystemInstanceId::new(
        SubsystemId::new("child").unwrap(),
        ReplicaIndex::ZERO,
    ));
    let module = ModuleId::new("worker").unwrap();
    let timeline = vec![
        EvalEvent {
            sequence: 1,
            offset_ms: 110,
            scope: root,
            module: module.clone(),
            replica: 0,
            step: None,
            payload: EvalEventPayload::MemoUpdated { char_count: 1 },
        },
        EvalEvent {
            sequence: 2,
            offset_ms: 180,
            scope: child,
            module,
            replica: 0,
            step: None,
            payload: EvalEventPayload::MemoUpdated { char_count: 1 },
        },
    ];
    let selector = EventSelector::default();
    assert_eq!(
        measure::first_match_latency_ms(&timeline, &selector, 100, true),
        measure::MeasurementValue::ByScope(
            [("/".into(), 10.0), ("/child[0]".into(), 80.0)]
                .into_iter()
                .collect()
        )
    );
    assert_eq!(
        measure::scope_coverage(
            &timeline,
            &selector,
            &BTreeSet::from(["/".into(), "/child[0]".into()]),
        ),
        measure::MeasurementValue::Scalar(Some(1.0))
    );
    assert_eq!(
        measure::scope_convergence_latency_ms(
            &timeline,
            &selector,
            &BTreeSet::from(["/".into(), "/child[0]".into()]),
            100,
        ),
        measure::MeasurementValue::Scalar(Some(80.0))
    );
}

#[test]
fn scope_coverage_can_measure_cognition_origin_instead_of_storage_scope() {
    let root = ScopeId::root();
    let module = ModuleId::new("cognition-gate").unwrap();
    let timeline = [
        (1, "/left-leg[0]/query-memory"),
        (2, "/center-leg[0]/query-memory"),
        (3, "/right-leg[0]/query-memory"),
    ]
    .into_iter()
    .map(|(sequence, origin)| EvalEvent {
        sequence,
        offset_ms: sequence * 10,
        scope: root.clone(),
        module: module.clone(),
        replica: 0,
        step: Some("retrieve-key".into()),
        payload: EvalEventPayload::CognitionAppended {
            content: "shared fact".into(),
            origin: origin.into(),
        },
    })
    .collect::<Vec<_>>();
    let expected = BTreeSet::from([
        "/left-leg[0]".into(),
        "/center-leg[0]".into(),
        "/right-leg[0]".into(),
    ]);
    let selector = EventSelector {
        scopes: ScopeSelector::Exact(BTreeSet::from(["/".into()])),
        origin_scopes: expected.clone(),
        modules: BTreeSet::from(["cognition-gate".into()]),
        variants: BTreeSet::from(["cognition-appended".into()]),
        steps: BTreeSet::from(["retrieve-key".into()]),
        ..EventSelector::default()
    };

    assert_eq!(
        measure::scope_coverage(&timeline, &selector, &expected),
        measure::MeasurementValue::Scalar(Some(1.0))
    );
}

#[test]
fn missing_latency_is_reported_as_null() {
    let value = measure::first_match_latency_ms(&[], &EventSelector::default(), 0, false);
    assert_eq!(
        serde_json::to_value(value).expect("serialize missing measurement"),
        serde_json::Value::Null
    );
}
