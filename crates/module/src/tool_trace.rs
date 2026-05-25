pub fn emit_trace_tool_calls<C>(calls: &[C])
where
    C: lutum::ToolCallWrapper,
{
    for call in calls {
        let metadata = call.metadata();
        tracing::event!(
            target: "lutum",
            tracing::Level::DEBUG,
            tool_name = metadata.name.as_str(),
            args_json = metadata.arguments.get(),
            tool_call_id = metadata.id.as_str(),
            "tool_call"
        );
    }
}
