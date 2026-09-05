use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    cases::{EventSelectorSpec, Measurement},
    query::{EventSelector, ScopeSelector},
    timeline::EvalEvent,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeasurementValue {
    Scalar(Option<f64>),
    ByScope(BTreeMap<String, f64>),
}

pub fn count(timeline: &[EvalEvent], selector: &EventSelector) -> MeasurementValue {
    MeasurementValue::Scalar(Some(selector.select(timeline).len() as f64))
}

pub fn first_match_latency_ms(
    timeline: &[EvalEvent],
    selector: &EventSelector,
    anchor_ms: u64,
    group_by_scope: bool,
) -> MeasurementValue {
    let selected = selector.select(timeline);
    if group_by_scope {
        let mut values = BTreeMap::new();
        for event in selected {
            let scope = selector
                .measurement_scope(event)
                .map_or_else(|| event.scope.to_string(), str::to_owned);
            values
                .entry(scope)
                .or_insert_with(|| event.offset_ms.saturating_sub(anchor_ms) as f64);
        }
        MeasurementValue::ByScope(values)
    } else {
        MeasurementValue::Scalar(
            selected
                .first()
                .map(|event| event.offset_ms.saturating_sub(anchor_ms) as f64),
        )
    }
}

pub fn unique_scope_count(timeline: &[EvalEvent], selector: &EventSelector) -> MeasurementValue {
    let scopes = selector
        .select(timeline)
        .into_iter()
        .map(|event| {
            selector
                .measurement_scope(event)
                .map_or_else(|| event.scope.to_string(), str::to_owned)
        })
        .collect::<BTreeSet<_>>();
    MeasurementValue::Scalar(Some(scopes.len() as f64))
}

pub fn scope_coverage(
    timeline: &[EvalEvent],
    selector: &EventSelector,
    expected_scopes: &BTreeSet<String>,
) -> MeasurementValue {
    if expected_scopes.is_empty() {
        return MeasurementValue::Scalar(None);
    }
    let observed = selector
        .select(timeline)
        .into_iter()
        .map(|event| {
            selector
                .measurement_scope(event)
                .map_or_else(|| event.scope.to_string(), str::to_owned)
        })
        .collect::<BTreeSet<_>>();
    let covered = expected_scopes.intersection(&observed).count();
    MeasurementValue::Scalar(Some(covered as f64 / expected_scopes.len() as f64))
}

pub fn scope_convergence_latency_ms(
    timeline: &[EvalEvent],
    selector: &EventSelector,
    expected_scopes: &BTreeSet<String>,
    anchor_ms: u64,
) -> MeasurementValue {
    if expected_scopes.is_empty() {
        return MeasurementValue::Scalar(None);
    }
    let first_by_scope = selector
        .select(timeline)
        .into_iter()
        .filter_map(|event| {
            let scope = selector
                .measurement_scope(event)
                .map_or_else(|| event.scope.to_string(), str::to_owned);
            expected_scopes.contains(&scope).then_some((scope, event))
        })
        .fold(BTreeMap::new(), |mut first, event| {
            let (scope, event) = event;
            first.entry(scope).or_insert(event.offset_ms);
            first
        });
    if first_by_scope.len() != expected_scopes.len() {
        return MeasurementValue::Scalar(None);
    }
    MeasurementValue::Scalar(
        first_by_scope
            .values()
            .max()
            .map(|offset| offset.saturating_sub(anchor_ms) as f64),
    )
}

pub fn evaluate_declared(
    timeline: &[EvalEvent],
    measurements: &[Measurement],
) -> BTreeMap<String, MeasurementValue> {
    measurements
        .iter()
        .map(|measurement| {
            let (name, value) = match measurement {
                Measurement::Count { name, select } => (name, count(timeline, &selector(select))),
                Measurement::FirstMatchLatency {
                    name,
                    select,
                    group_by_scope,
                } => (
                    name,
                    first_match_latency_ms(
                        timeline,
                        &selector(select),
                        measurement_anchor_ms(timeline, select),
                        *group_by_scope,
                    ),
                ),
                Measurement::UniqueScopeCount { name, select } => {
                    (name, unique_scope_count(timeline, &selector(select)))
                }
                Measurement::ScopeCoverage {
                    name,
                    select,
                    scopes,
                } => (
                    name,
                    scope_coverage(
                        timeline,
                        &selector(select),
                        &scopes.iter().cloned().collect(),
                    ),
                ),
                Measurement::ScopeConvergenceLatency {
                    name,
                    select,
                    scopes,
                } => (
                    name,
                    scope_convergence_latency_ms(
                        timeline,
                        &selector(select),
                        &scopes.iter().cloned().collect(),
                        measurement_anchor_ms(timeline, select),
                    ),
                ),
            };
            (name.clone(), value)
        })
        .collect()
}

fn measurement_anchor_ms(timeline: &[EvalEvent], spec: &EventSelectorSpec) -> u64 {
    if spec.steps.len() != 1 {
        return 0;
    }
    timeline
        .iter()
        .find(|event| {
            event.payload.variant() == "stimulus-published"
                && event.step.as_deref() == Some(spec.steps[0].as_str())
        })
        .map_or(0, |event| event.offset_ms)
}

fn selector(spec: &EventSelectorSpec) -> EventSelector {
    EventSelector {
        scopes: if spec.scopes.is_empty() {
            ScopeSelector::Any
        } else {
            ScopeSelector::Exact(spec.scopes.iter().cloned().collect())
        },
        origin_scopes: spec.origin_scopes.iter().cloned().collect(),
        modules: spec
            .modules
            .iter()
            .map(|module| module.as_str().to_owned())
            .collect(),
        replicas: spec.replicas.iter().copied().collect(),
        variants: spec.variants.iter().cloned().collect(),
        steps: spec.steps.iter().cloned().collect(),
    }
}
