import os
import asyncio
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Укажите ваш API ключ
API_KEY = "AIzaSyDhJGPPleCkBzYklQlTBTmLazy9TmkgI_4"

# Инициализация модели Google Generative AI (Gemini)
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=API_KEY)


async def search_internet(prompt: str, max_steps: int = 15, max_actions_per_step: int = 6,
                          use_vision: bool = False) -> str:
    """
    Выполняет поиск данных в интернете по заданному промту.

    Аргументы:
        prompt (str): Текст запроса для агента.
        max_steps (int, optional): Максимальное количество шагов для агента. По умолчанию 15.
        max_actions_per_step (int, optional): Максимальное количество действий на шаг. По умолчанию 6.
        use_vision (bool, optional): Флаг использования визуальных возможностей. По умолчанию False.

    Возвращает:
        str: Результат работы агента в виде строки.
    """
    # Создаем агента с заданной задачей
    agent = Agent(
        task=prompt,
        llm=llm,
        max_actions_per_step=max_actions_per_step,
        use_vision=use_vision
    )

    # Запускаем агента с указанным количеством шагов
    result = await agent.run(max_steps=max_steps)
    return str(result)