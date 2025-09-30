import os
from utils.models import Task
from prompts.story.plan.plan_output import PlanOutput, convert_plan_to_tasks
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.base import hybrid_query_react
from story.story_rag import get_story_rag


async def hierarchy_proposer(task: Task) -> Task:
    proposer_system_prompt, proposer_user_prompt = load_prompts(f"prompts.{task.category}.hierarchy.hierarchy_1", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    proposer_messages = get_llm_messages(proposer_system_prompt, proposer_user_prompt, None, context)
    proposer_llm_params = get_llm_params(messages=proposer_messages, temperature=llm_temperatures["creative"])
    proposer_message = await llm_completion(proposer_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["hierarchy_proposer"] = proposer_message.content
    updated_task.results["hierarchy_proposer_reasoning"] = proposer_message.get("reasoning_content") or proposer_message.get("reasoning", "")
    return updated_task


async def hierarchy_proposer_react(task: Task) -> Task:
    """
    使用 ReAct 模式为层级分解任务执行提议步骤。
    Agent 可以在执行过程中动态查询所需信息。
    """
    # 1. 加载为 ReAct Agent 专门设计的 Prompt
    proposer_system_prompt, proposer_user_prompt = load_prompts(f"prompts.{task.category}.hierarchy.hierarchy_1_react", "system_prompt", "user_prompt")
    
    # 2. 仅获取基础上下文信息, 用于填充初始的用户提示
    context = get_story_rag().get_context_base(task)
    messages = get_llm_messages(proposer_system_prompt, proposer_user_prompt, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]

    # 3. 调用 ReAct Agent, 它内部封装了工具和思考-行动循环
    proposer_content = await hybrid_query_react(
        run_id=task.run_id,
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
    )
    updated_task = task.model_copy(deep=True)
    updated_task.results["hierarchy_proposer"] = proposer_content
    # 注意: ReAct 的完整思考链目前未捕获为 reasoning, 可以后续扩展 call_react_agent 函数来实现
    updated_task.results["hierarchy_proposer_reasoning"] = "ReAct agent executed."
    return updated_task


async def hierarchy_critic(task: Task) -> Task:
    critic_system_prompt, critic_user_prompt = load_prompts(f"prompts.{task.category}.hierarchy.hierarchy_2_critic", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    critic_context = context.copy()
    critic_context["proposer_draft"] = task.results.get("hierarchy_proposer")
    critic_messages = get_llm_messages(critic_system_prompt, critic_user_prompt, None, critic_context)
    critic_llm_params = get_llm_params(messages=critic_messages, temperature=llm_temperatures["reasoning"])
    critic_message = await llm_completion(critic_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["hierarchy_critic"] = critic_message.content
    updated_task.results["hierarchy_critic_reasoning"] = critic_message.get("reasoning_content") or critic_message.get("reasoning", "")
    return updated_task


async def hierarchy_synthesizer(task: Task) -> Task:
    synthesizer_system_prompt, synthesizer_user_prompt = load_prompts(f"prompts.{task.category}.hierarchy.hierarchy_3_synthesizer", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    synthesizer_context = context.copy()
    synthesizer_context.update({
        "proposer_draft": task.results.get("hierarchy_proposer"),
        "critic_feedback": task.results.get("hierarchy_critic"),
    })
    synthesizer_messages = get_llm_messages(synthesizer_system_prompt, synthesizer_user_prompt, None, synthesizer_context)
    synthesizer_llm_params = get_llm_params(messages=synthesizer_messages, temperature=llm_temperatures["reasoning"])
    final_message = await llm_completion(synthesizer_llm_params, response_model=PlanOutput)
    updated_task = task.model_copy(deep=True)
    data = final_message.validated_data
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["hierarchy"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    all_reasoning = [
        f"### Proposer Reasoning\n{task.results.get('hierarchy_proposer_reasoning', '')}",
        f"### Critic Reasoning\n{task.results.get('hierarchy_critic_reasoning', '')}",
        f"### Synthesizer Reasoning\n{final_message.get('reasoning_content') or final_message.get('reasoning', '')}",
        f"### Final Plan Reasoning\n{data.reasoning}"
    ]
    updated_task.results["hierarchy_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task

