import os
from utils.models import Task
from utils.llm import get_llm_messages, get_llm_params, llm_completion, llm_temperatures
from utils.loader import load_prompts
from story.story_rag import get_story_rag


async def translation_proposer(task: Task) -> Task:
    proposer_system_prompt, proposer_user_prompt = load_prompts(f"prompts.{task.category}.translation.translation_1", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    context["chinese_text"] = task.results.get("write")
    proposer_messages = get_llm_messages(proposer_system_prompt, proposer_user_prompt, None, context)
    proposer_llm_params = get_llm_params(messages=proposer_messages, temperature=llm_temperatures["creative"])
    proposer_message = await llm_completion(proposer_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["translation_text"] = proposer_message.content
    updated_task.results["translation_text_reasoning"] = proposer_message.get("reasoning_content") or proposer_message.get("reasoning", "")
    return updated_task


async def translation_critic(task: Task) -> Task:
    critic_system_prompt, critic_user_prompt = load_prompts(f"prompts.{task.category}.translation.translation_2_critic", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    critic_context = context.copy()
    critic_context["translation_text"] = task.results.get("translation_text")
    critic_messages = get_llm_messages(critic_system_prompt, critic_user_prompt, None, critic_context)
    critic_llm_params = get_llm_params(messages=critic_messages, temperature=llm_temperatures["reasoning"])
    critic_message = await llm_completion(critic_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["translation_critic"] = critic_message.content
    updated_task.results["translation_critic_reasoning"] = critic_message.get("reasoning_content") or critic_message.get("reasoning", "")
    return updated_task


async def translation_refine(task: Task) -> Task:
    refine_system_prompt, refine_user_prompt = load_prompts(f"prompts.{task.category}.translation.translation_3_refine", "system_prompt", "user_prompt")
    context = await get_story_rag().get_context(task)
    refine_context = context.copy()
    refine_context.update({
        "translation_text": task.results.get("translation_text"),
        "translation_critic": task.results.get("translation_critic"),
    })
    refine_messages = get_llm_messages(refine_system_prompt, refine_user_prompt, None, refine_context)
    refine_llm_params = get_llm_params(messages=refine_messages, temperature=llm_temperatures["reasoning"])
    final_message = await llm_completion(refine_llm_params)
    updated_task = task.model_copy(deep=True)
    updated_task.results["translation"] = final_message.content
    all_reasoning = [
        f"### Proposer Reasoning\n{task.results.get('translation_text_reasoning', '')}",
        f"### Critic Reasoning\n{task.results.get('translation_critic_reasoning', '')}",
        f"### Refine Reasoning\n{final_message.get('reasoning_content') or final_message.get('reasoning', '')}",
    ]
    updated_task.results["translation_reasoning"] = "\n\n".join(filter(None, all_reasoning))
    return updated_task
