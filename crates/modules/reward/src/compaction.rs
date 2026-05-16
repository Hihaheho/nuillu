use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lutum::{Session, TextStepOutcomeWithTools, ToolResult};
use nuillu_module::{
    AllocationReader, AllocationUpdatedInbox, BlackboardReader, LlmAccess, Module,
};
use nuillu_types::{ModuleId, PolicyIndex, PolicyRank};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{PolicyCompactor, PolicyRecord, PolicySearchHit};

const SYSTEM_PROMPT: &str = r#"You are the policy-compaction module.
Inspect durable policies and conservatively remove redundant policy duplicates. Use
compact_duplicate_policies only when duplicate policies are already covered by one canonical policy's
trigger and behavior without contradiction. Keep separate policies when they differ in behavior,
scope, valence, risk, or useful specificity. Never delete Core policies. Do not insert, reinforce,
or rewrite policy values."#;

#[lutum::tool_input(name = "search_policies", output = SearchPoliciesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchPoliciesArgs {
    pub trigger: String,
    pub limit: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchPoliciesOutput {
    pub policies: Vec<PolicyContentView>,
}

#[lutum::tool_input(name = "get_policies", output = GetPoliciesOutput)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetPoliciesArgs {
    pub indexes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GetPoliciesOutput {
    pub policies: Vec<PolicyContentView>,
}

#[lutum::tool_input(
    name = "compact_duplicate_policies",
    output = CompactDuplicatePoliciesOutput
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CompactDuplicatePoliciesArgs {
    pub canonical_index: String,
    pub duplicate_indexes: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CompactDuplicatePoliciesOutput {
    pub canonical_index: String,
    pub deleted_indexes: Vec<String>,
    pub skipped: Vec<PolicyCompactionSkippedView>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCompactionSkippedView {
    pub index: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyContentView {
    pub index: String,
    pub rank: PolicyRank,
    pub trigger: String,
    pub behavior: String,
    pub expected_reward: f32,
    pub confidence: f32,
    pub value: f32,
    pub reward_tokens: u32,
    pub reinforcement_count: u32,
    pub decay_remaining_secs: i64,
    pub similarity: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum PolicyCompactionTools {
    SearchPolicies(SearchPoliciesArgs),
    GetPolicies(GetPoliciesArgs),
    CompactDuplicatePolicies(CompactDuplicatePoliciesArgs),
}

pub struct PolicyCompactionModule {
    owner: ModuleId,
    allocation_updates: AllocationUpdatedInbox,
    allocation: AllocationReader,
    blackboard: BlackboardReader,
    compactor: PolicyCompactor,
    llm: LlmAccess,
    system_prompt: std::sync::OnceLock<String>,
}

impl PolicyCompactionModule {
    pub fn new(
        allocation_updates: AllocationUpdatedInbox,
        allocation: AllocationReader,
        blackboard: BlackboardReader,
        compactor: PolicyCompactor,
        llm: LlmAccess,
    ) -> Self {
        Self {
            owner: ModuleId::new(<Self as Module>::id()).expect("policy-compaction id is valid"),
            allocation_updates,
            allocation,
            blackboard,
            compactor,
            llm,
            system_prompt: std::sync::OnceLock::new(),
        }
    }

    fn system_prompt(&self, cx: &nuillu_module::ActivateCx<'_>) -> &str {
        self.system_prompt.get_or_init(|| {
            nuillu_module::format_system_prompt(
                SYSTEM_PROMPT,
                cx.modules(),
                &self.owner,
                cx.identity_memories(),
                cx.core_policies(),
                cx.now(),
            )
        })
    }

    #[tracing::instrument(skip_all, err(Debug, level = "warn"))]
    async fn activate(&self, cx: &nuillu_module::ActivateCx<'_>) -> Result<()> {
        let policies = self
            .compactor
            .list_compaction_candidates()
            .await
            .context("list policy compaction candidates")?;
        let reinforcement_counts = self.reinforcement_counts().await;
        let policy_views = policies
            .into_iter()
            .map(|record| policy_record_to_view(record, &reinforcement_counts, None))
            .collect::<Vec<_>>();
        let allocation = self.allocation.snapshot().await;
        let allocation_guidance = allocation
            .iter()
            .filter_map(|(id, config)| {
                let guidance = config.guidance.trim();
                (!guidance.is_empty()).then(|| (id.to_string(), guidance.to_owned()))
            })
            .collect::<Vec<_>>();

        let mut session = Session::new();
        session.push_system(self.system_prompt(cx));
        session.push_user(format_policy_compaction_context(
            &policy_views,
            &allocation_guidance,
        ));

        for _ in 0..6 {
            let lutum = self.llm.lutum().await;
            let outcome = session
                .text_turn(&lutum)
                .tools::<PolicyCompactionTools>()
                .available_tools([
                    PolicyCompactionToolsSelector::SearchPolicies,
                    PolicyCompactionToolsSelector::GetPolicies,
                    PolicyCompactionToolsSelector::CompactDuplicatePolicies,
                ])
                .collect()
                .await
                .context("policy-compaction text turn failed")?;

            match outcome {
                TextStepOutcomeWithTools::Finished(_) => return Ok(()),
                TextStepOutcomeWithTools::FinishedNoOutput(_) => return Ok(()),
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    if round.tool_calls.is_empty() {
                        tracing::warn!("policy-compaction requested tools without tool calls");
                        return Ok(());
                    }
                    let mut results: Vec<ToolResult> = Vec::new();
                    for call in round.tool_calls.iter().cloned() {
                        match call {
                            PolicyCompactionToolsCall::SearchPolicies(call) => {
                                let output = self
                                    .search_policies(call.input.clone())
                                    .await
                                    .context("run search_policies tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete search_policies tool call")?;
                                results.push(result);
                            }
                            PolicyCompactionToolsCall::GetPolicies(call) => {
                                let output = self
                                    .get_policies(call.input.clone())
                                    .await
                                    .context("run get_policies tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete get_policies tool call")?;
                                results.push(result);
                            }
                            PolicyCompactionToolsCall::CompactDuplicatePolicies(call) => {
                                let output = self
                                    .compact_duplicate_policies(call.input.clone())
                                    .await
                                    .context("run compact_duplicate_policies tool")?;
                                let result = call
                                    .complete(output)
                                    .context("complete compact_duplicate_policies tool call")?;
                                results.push(result);
                            }
                        }
                    }
                    round
                        .commit(&mut session, results)
                        .context("commit policy-compaction tool round")?;
                }
            }
        }
        Ok(())
    }

    async fn search_policies(&self, args: SearchPoliciesArgs) -> Result<SearchPoliciesOutput> {
        let hits = self
            .compactor
            .search(&args.trigger, args.limit)
            .await
            .context("search policies")?;
        Ok(SearchPoliciesOutput {
            policies: self.hit_views(hits).await,
        })
    }

    async fn get_policies(&self, args: GetPoliciesArgs) -> Result<GetPoliciesOutput> {
        let indexes = args
            .indexes
            .into_iter()
            .map(PolicyIndex::new)
            .collect::<Vec<_>>();
        let records = self
            .compactor
            .get_many(&indexes)
            .await
            .context("get policies")?;
        let counts = self.reinforcement_counts().await;
        Ok(GetPoliciesOutput {
            policies: records
                .into_iter()
                .map(|record| policy_record_to_view(record, &counts, None))
                .collect(),
        })
    }

    async fn compact_duplicate_policies(
        &self,
        args: CompactDuplicatePoliciesArgs,
    ) -> Result<CompactDuplicatePoliciesOutput> {
        let canonical = PolicyIndex::new(args.canonical_index);
        let duplicates = args
            .duplicate_indexes
            .into_iter()
            .map(PolicyIndex::new)
            .collect::<Vec<_>>();
        let result = self
            .compactor
            .compact_duplicates(&canonical, &duplicates)
            .await
            .context("compact duplicate policies")?;
        Ok(CompactDuplicatePoliciesOutput {
            canonical_index: result.canonical.index.to_string(),
            deleted_indexes: result
                .deleted
                .into_iter()
                .map(|index| index.to_string())
                .collect(),
            skipped: result
                .skipped
                .into_iter()
                .map(|skipped| PolicyCompactionSkippedView {
                    index: skipped.index.to_string(),
                    reason: skipped.reason,
                })
                .collect(),
        })
    }

    async fn hit_views(&self, hits: Vec<PolicySearchHit>) -> Vec<PolicyContentView> {
        let counts = self.reinforcement_counts().await;
        hits.into_iter()
            .map(|hit| policy_record_to_view(hit.policy, &counts, Some(hit.similarity)))
            .collect()
    }

    async fn reinforcement_counts(&self) -> HashMap<PolicyIndex, u32> {
        self.blackboard
            .read(|bb| {
                bb.policy_metadata()
                    .iter()
                    .map(|(index, metadata)| (index.clone(), metadata.reinforcement_count))
                    .collect()
            })
            .await
    }

    async fn next_batch(&mut self) -> Result<()> {
        let _ = self.allocation_updates.next_item().await?;
        let _ = self.allocation_updates.take_ready_items()?;
        Ok(())
    }
}

fn policy_record_to_view(
    record: PolicyRecord,
    reinforcement_counts: &HashMap<PolicyIndex, u32>,
    similarity: Option<f32>,
) -> PolicyContentView {
    let reinforcement_count = reinforcement_counts
        .get(&record.index)
        .copied()
        .unwrap_or_default();
    PolicyContentView {
        index: record.index.to_string(),
        rank: record.rank,
        trigger: record.trigger,
        behavior: record.behavior,
        expected_reward: record.expected_reward.get(),
        confidence: record.confidence.get(),
        value: record.value.get(),
        reward_tokens: record.reward_tokens,
        reinforcement_count,
        decay_remaining_secs: record.decay_remaining_secs,
        similarity,
    }
}

fn format_policy_compaction_context(
    policies: &[PolicyContentView],
    allocation_guidance: &[(String, String)],
) -> String {
    let mut out = String::from("Policy compaction context.");
    out.push_str("\n\nPolicy candidates:");
    if policies.is_empty() {
        out.push_str("\n- none");
    } else {
        for policy in policies {
            out.push_str(&format!(
                "\n- {}: rank={:?}; expected_reward={:.3}; confidence={:.3}; value={:.3}; reward_tokens={}; reinforcement_count={}; decay_remaining_secs={}; trigger={:?}; behavior={:?}",
                policy.index,
                policy.rank,
                policy.expected_reward,
                policy.confidence,
                policy.value,
                policy.reward_tokens,
                policy.reinforcement_count,
                policy.decay_remaining_secs,
                policy.trigger,
                policy.behavior,
            ));
        }
    }
    out.push_str("\n\nCurrent compaction guidance:");
    if allocation_guidance.is_empty() {
        out.push_str("\n- none");
    } else {
        for (id, guidance) in allocation_guidance {
            out.push_str(&format!("\n- {id}: {guidance}"));
        }
    }
    out
}

#[async_trait(?Send)]
impl Module for PolicyCompactionModule {
    type Batch = ();

    fn id() -> &'static str {
        "policy-compaction"
    }

    fn role_description() -> &'static str {
        "Conservatively deletes redundant non-Core policy duplicates without inserting, reinforcing, or rewriting policies."
    }

    async fn next_batch(&mut self) -> Result<Self::Batch> {
        PolicyCompactionModule::next_batch(self).await
    }

    async fn activate(
        &mut self,
        cx: &nuillu_module::ActivateCx<'_>,
        _batch: &Self::Batch,
    ) -> Result<()> {
        PolicyCompactionModule::activate(self, cx).await
    }
}
