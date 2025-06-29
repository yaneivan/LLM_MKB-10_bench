import os
import csv
import argparse
import re
import glob
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

# ==============================================================================
# НАСТРОЙКИ API И МОДЕЛИ (теперь они в основном для отчетности)
# ==============================================================================
BASE_URL = "http://10.203.1.11:4242/v1"
MODEL_NAME = "google/gemma-3-4b-it" # Указываем имя модели, которую тестируем
TEMPERATURE = 0.0
RESULTS_FILE = "test_results.csv"
CONCURRENT_REQUESTS = 30000 # Количество одновременных запросов к vLLM

# ==============================================================================

# ... (Код для update_results_matrix_csv остается тем же) ...
def update_results_matrix_csv(model_name: str, benchmark_name: str, accuracy: str):
    data = {}
    header = ["model_name"]
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames if reader.fieldnames else ["model_name"]
            for row in reader: data[row['model_name']] = {h: row.get(h, '') for h in header if h != 'model_name'}
    if benchmark_name not in header: header.append(benchmark_name)
    if model_name not in data: data[model_name] = {}
    data[model_name][benchmark_name] = accuracy
    all_benchmarks = set(h for h in header if h != 'model_name')
    for model_results in data.values(): all_benchmarks.update(model_results.keys())
    final_header = ["model_name"] + sorted(list(all_benchmarks))
    with open(RESULTS_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_header)
        writer.writeheader()
        for model, results in sorted(data.items()):
            row_to_write = {'model_name': model}; row_to_write.update(results); writer.writerow(row_to_write)


async def fetch_one(session, semaphore, prompt_data):
    """Асинхронно выполняет один запрос к API."""
    async with semaphore:
        system_prompt = "Твоя задача — ответить на вопрос с вариантами ответа. В качестве ответа дай ТОЛЬКО одну букву (A, B, C или D), соответствующую правильному варианту. Не добавляй никаких объяснений или лишних слов."
        user_prompt = prompt_data["prompt"]
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": TEMPERATURE, "max_tokens": 5,
        }
        
        try:
            async with session.post(f"{BASE_URL.rstrip('/')}/chat/completions", json=payload, timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["choices"][0]["message"]["content"]
                    parsed_answer = re.search(r'^\s*([A-D])', response_text.strip(), re.IGNORECASE)
                    if parsed_answer and parsed_answer.group(1).upper() == prompt_data["correct_letter"]:
                        return "correct"
                    elif parsed_answer:
                        return "incorrect"
                    else:
                        return "parsing_failure"
                else:
                    return "api_error"
        except Exception:
            return "api_error"

async def run_single_test_async(benchmark_path: str):
    """Запускает асинхронное тестирование на одном бенчмарке."""
    if not os.path.exists(benchmark_path):
        print(f"Ошибка: Файл бенчмарка не найден: {benchmark_path}"); return

    with open(benchmark_path, mode='r', encoding='utf-8') as infile:
        questions = list(csv.DictReader(infile))

    # Подготовка всех запросов заранее
    all_prompts = []
    for row in questions:
        correct_answer_letter, correct_choice_text = row['answer'], row[row['answer']]
        distractors = [row[l] for l in ['A', 'B', 'C', 'D'] if l != correct_answer_letter]
        for i in range(4):
            choices = list(distractors); choices.insert(i, correct_choice_text)
            prompt = f"Вопрос: {row['question']}\n\nВарианты ответа:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
            all_prompts.append({"prompt": prompt, "correct_letter": chr(ord('A') + i)})

    # Асинхронное выполнение
    benchmark_name = os.path.basename(benchmark_path)
    print(f"\n--- Начинаем асинхронный тест на '{benchmark_name}' ({len(all_prompts)} запросов) ---")
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, semaphore, p) for p in all_prompts]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Тестируем {benchmark_name}")

    # Подсчет результатов
    correct = results.count("correct")
    parsing_failures = results.count("parsing_failure")
    api_errors = results.count("api_error")
    successful_attempts = len(all_prompts) - api_errors - parsing_failures
    accuracy = (correct / successful_attempts) * 100 if successful_attempts > 0 else 0.0
    accuracy_str = f"{accuracy:.2f}%"

    update_results_matrix_csv(MODEL_NAME, benchmark_name, accuracy_str)
    print("--- Результаты теста ---")
    print(f"  Точность: {accuracy_str} ({correct}/{successful_attempts}) | Ошибки парсинга: {parsing_failures}, API: {api_errors}")
    print(f"Сводная таблица обновлена в файле '{RESULTS_FILE}'")
    print("------------------------")

if __name__ == "__main__":
    # Установка зависимостей для этого скрипта: pip install aiohttp tqdm
    parser = argparse.ArgumentParser(description="Асинхронный скрипт для тестирования LLM на бенчмарках МКБ-10.")
    parser.add_argument('--benchmark-file', type=str, nargs='?', default=None, help="Путь к CSV файлу. Если не указан, запускаются все тесты.")
    args = parser.parse_args()

    files_to_run = [args.benchmark_file] if args.benchmark_file else glob.glob("mkb10_benchmark_*.csv")
    if not files_to_run or files_to_run == [None]:
        print("Не найдено файлов бенчмарка. Сгенерируйте их с помощью main.py.")
        exit()
    
    print(f"Найдено {len(files_to_run)} файлов для тестирования.")
    
    async def main():
        for benchmark_file in sorted(files_to_run):
            await run_single_test_async(benchmark_file)

    asyncio.run(main())
    print("\nВсе тесты завершены.")