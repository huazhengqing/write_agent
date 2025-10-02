from prompts.story.plan_output import PlanOutput, convert_plan_to_tasks
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag
from story.base import hybrid_query_react



async def plan_write_proposer(task: Task) -> Task:
    """为 write_to_design 任务执行提议步骤"""
    context = await get_story_rag().get_context(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", ""),
        "complex_reasons": task.results.get("complex_reasons") or ""
    })
    proposer_system, proposer_user = load_prompts(f"prompts.{task.category}.plan.plan_write_1", "system_prompt", "user_prompt")
    proposer_messages = get_llm_messages(proposer_system, proposer_user, None, context)
    proposer_llm_params = get_llm_params(messages=proposer_messages, temperature=llm_temperatures["creative"])
    proposer_message = await llm_completion(proposer_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_proposer"] = proposer_message.content
    updated_task.results["plan_proposer_reasoning"] = proposer_message.get("reasoning_content") or proposer_message.get("reasoning", "")
    return updated_task


async def plan_write_proposer_react(task: Task) -> Task:
    """为 write_to_design 任务使用 ReAct 模式执行提议步骤"""
    # 1. 加载为 ReAct Agent 专门设计的 Prompt
    proposer_system, proposer_user = load_prompts(f"prompts.{task.category}.plan.plan_write_1_react", "system_prompt", "user_prompt")

    # 2. 仅获取基础上下文信息
    context = get_story_rag().get_context_base(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", ""),
        "complex_reasons": task.results.get("complex_reasons") or ""
    })
    messages = get_llm_messages(proposer_system, proposer_user, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]

    # 3. 调用 ReAct Agent
    proposer_content = await hybrid_query_react(
        run_id=task.run_id,
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
    )
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_proposer"] = proposer_content
    updated_task.results["plan_proposer_reasoning"] = "ReAct agent for plan_write_proposer executed."
    return updated_task


async def plan_write_critic(task: Task) -> Task:
    """为 write_to_design 任务执行批判步骤"""
    context = await get_story_rag().get_context(task)
    critic_context = context.copy()
    critic_context["proposer_ideas"] = task.results.get("plan_proposer")
    critic_system, critic_user = load_prompts(f"prompts.{task.category}.plan.plan_write_2_critic", "system_prompt", "user_prompt")
    critic_messages = get_llm_messages(critic_system, critic_user, None, critic_context)
    critic_llm_params = get_llm_params(messages=critic_messages, temperature=llm_temperatures["reasoning"])
    critic_message = await llm_completion(critic_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_critic"] = critic_message.content
    updated_task.results["plan_critic_reasoning"] = critic_message.get("reasoning_content") or critic_message.get("reasoning", "")
    return updated_task


async def plan_write_synthesizer(task: Task) -> Task:
    """为 write_to_design 任务执行整合步骤"""
    context = await get_story_rag().get_context(task)
    synthesizer_context = context.copy()
    synthesizer_context["draft_plan"] = task.results.get("plan_critic")
    synthesizer_system, synthesizer_user = load_prompts(f"prompts.{task.category}.plan.plan_write_3_synthesizer", "system_prompt", "user_prompt")
    synthesizer_messages = get_llm_messages(synthesizer_system, synthesizer_user, None, synthesizer_context)
    synthesizer_llm_params = get_llm_params(messages=synthesizer_messages, temperature=llm_temperatures["reasoning"])
    final_message = await llm_completion(synthesizer_llm_params, response_model=PlanOutput)
    data = final_message.validated_data
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["plan"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    all_reasoning = [
        f"### Proposer Reasoning\n{task.results.get('plan_proposer_reasoning', '')}",
        f"### Critic Reasoning\n{task.results.get('plan_critic_reasoning', '')}",
        f"### Synthesizer Reasoning\n{final_message.get('reasoning_content') or final_message.get('reasoning', '')}",
        f"### Final Plan Reasoning\n{data.reasoning}"
    ]
    updated_task.results["plan_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task



###############################################################################



async def plan_design_proposer(task: Task) -> Task:
    """为 design 任务执行提议步骤"""
    context = await get_story_rag().get_context(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", ""),
        "complex_reasons": task.results.get("complex_reasons") or ""
    })
    proposer_system, proposer_user = load_prompts(f"prompts.{task.category}.plan.plan_design_1", "system_prompt", "user_prompt")
    proposer_messages = get_llm_messages(proposer_system, proposer_user, None, context)
    proposer_llm_params = get_llm_params(messages=proposer_messages, temperature=llm_temperatures["creative"])
    proposer_message = await llm_completion(proposer_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_proposer"] = proposer_message.content
    updated_task.results["plan_proposer_reasoning"] = proposer_message.get("reasoning_content") or proposer_message.get("reasoning", "")
    return updated_task


async def plan_design_proposer_react(task: Task) -> Task:
    """为 design 任务使用 ReAct 模式执行提议步骤"""
    # 1. 加载为 ReAct Agent 专门设计的 Prompt
    proposer_system, proposer_user = load_prompts(f"prompts.{task.category}.plan.plan_design_1_react", "system_prompt", "user_prompt")

    # 2. 仅获取基础上下文信息
    context = get_story_rag().get_context_base(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", ""),
        "complex_reasons": task.results.get("complex_reasons") or ""
    })
    messages = get_llm_messages(proposer_system, proposer_user, None, context)
    final_system_prompt = messages[0]["content"]
    final_user_prompt = messages[1]["content"]

    # 3. 调用 ReAct Agent
    proposer_content = await hybrid_query_react(
        run_id=task.run_id,
        system_prompt=final_system_prompt,
        user_prompt=final_user_prompt,
    )
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_proposer"] = proposer_content
    updated_task.results["plan_proposer_reasoning"] = "ReAct agent for plan_design_proposer executed."
    return updated_task


async def plan_design_critic(task: Task) -> Task:
    """为 design 任务执行批判步骤"""
    context = await get_story_rag().get_context(task)
    critic_context = context.copy()
    critic_context["proposer_ideas"] = task.results.get("plan_proposer")
    critic_system, critic_user = load_prompts(f"prompts.{task.category}.plan.plan_design_2_critic", "system_prompt", "user_prompt")
    critic_messages = get_llm_messages(critic_system, critic_user, None, critic_context)
    critic_llm_params = get_llm_params(messages=critic_messages, temperature=llm_temperatures["reasoning"])
    critic_message = await llm_completion(critic_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_critic"] = critic_message.content
    updated_task.results["plan_critic_reasoning"] = critic_message.get("reasoning_content") or critic_message.get("reasoning", "")
    return updated_task


async def plan_design_synthesizer(task: Task) -> Task:
    """为 design 任务执行整合步骤"""
    context = await get_story_rag().get_context(task)
    synthesizer_context = context.copy()
    synthesizer_context["draft_plan"] = task.results.get("plan_critic")
    synthesizer_system, synthesizer_user = load_prompts(f"prompts.{task.category}.plan.plan_design_3_synthesizer", "system_prompt", "user_prompt")
    synthesizer_messages = get_llm_messages(synthesizer_system, synthesizer_user, None, synthesizer_context)
    synthesizer_llm_params = get_llm_params(messages=synthesizer_messages, temperature=llm_temperatures["reasoning"])
    final_message = await llm_completion(synthesizer_llm_params, response_model=PlanOutput)
    data = final_message.validated_data
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["plan"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    all_reasoning = [
        f"### Proposer Reasoning\n{task.results.get('plan_proposer_reasoning', '')}",
        f"### Critic Reasoning\n{task.results.get('plan_critic_reasoning', '')}",
        f"### Synthesizer Reasoning\n{final_message.get('reasoning_content') or final_message.get('reasoning', '')}",
        f"### Final Plan Reasoning\n{data.reasoning}"
    ]
    updated_task.results["plan_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task



###############################################################################



async def plan_search_planner(task: Task) -> Task:
    """为 search 任务执行规划步骤"""
    context = get_story_rag().get_context_base(task)
    context.update({
        "atom_reasoning": task.results.get("atom_reasoning", ""),
        "complex_reasons": task.results.get("complex_reasons") or ""
    })
    planner_system, planner_user = load_prompts(f"prompts.{task.category}.plan.plan_search_1", "system_prompt", "user_prompt")
    planner_messages = get_llm_messages(planner_system, planner_user, None, context)
    planner_llm_params = get_llm_params(messages=planner_messages, temperature=llm_temperatures["reasoning"])
    planner_message = await llm_completion(planner_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["plan_planner"] = planner_message.content
    updated_task.results["plan_planner_reasoning"] = planner_message.get("reasoning_content") or planner_message.get("reasoning", "")
    return updated_task


async def plan_search_synthesizer(task: Task) -> Task:
    """为 search 任务执行整合步骤"""
    context = get_story_rag().get_context_base(task)
    synthesizer_context = context.copy()
    synthesizer_context["draft_plan"] = task.results.get("plan_planner")
    synthesizer_system, synthesizer_user = load_prompts(f"prompts.{task.category}.plan.plan_search_2_synthesizer", "system_prompt", "user_prompt")
    synthesizer_messages = get_llm_messages(synthesizer_system, synthesizer_user, None, synthesizer_context)
    synthesizer_llm_params = get_llm_params(messages=synthesizer_messages, temperature=llm_temperatures["reasoning"])
    final_message = await llm_completion(synthesizer_llm_params, response_model=PlanOutput)
    data = final_message.validated_data
    updated_task = task.model_copy(deep=True)
    updated_task.sub_tasks = convert_plan_to_tasks(data.sub_tasks, updated_task)
    updated_task.results["plan"] = data.model_dump(exclude_none=True, exclude={'reasoning'})
    all_reasoning = [
        f"### Planner Reasoning\n{task.results.get('plan_planner_reasoning', '')}",
        f"### Synthesizer Reasoning\n{final_message.get('reasoning_content') or final_message.get('reasoning', '')}",
        f"### Final Plan Reasoning\n{data.reasoning}"
    ]
    updated_task.results["plan_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task
