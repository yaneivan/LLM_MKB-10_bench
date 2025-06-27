import csv
import random
import re
import os
import argparse
from collections import defaultdict

class MKB10BenchmarkGenerator:
    """
    Расширяемый класс для генерации бенчмарков по справочнику МКБ-10.
    Поддерживает прямые (описание -> код) и обратные (код -> описание) задачи.
    """
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Файл не найден: {csv_path}. Убедитесь, что файл существует.")
        
        self.all_data, _ = self._load_data(csv_path)
        if not self.all_data: raise ValueError("Данные не загружены.")
        
        self.block_codes = [item for item in self.all_data if re.match(r'^[A-Z]\d{2}-[A-Z]\d{2}$', item['code'])]
        self.level1_codes = [item for item in self.all_data if re.match(r'^[A-Z]\d{2}$', item['code'])]
        self.sublevel_codes = [item for item in self.all_data if '.' in item['code']]

        print(f"Загружено {len(self.all_data)} строк. Найдено:")
        print(f"  - {len(self.block_codes)} кодов блоков, {len(self.level1_codes)} кодов 3-го уровня, {len(self.sublevel_codes)} кодов 4-го уровня.")

    def _load_data(self, csv_path: str):
        all_data, parent_child_map = [], defaultdict(list)
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                try:
                    item_id, _, code, desc, parent_id = row[0], row[1], row[2], row[3], row[4]
                    code_data = {'id': item_id, 'code': code.strip(), 'description': desc.strip(), 'parent_id': parent_id}
                    all_data.append(code_data)
                    if parent_id: parent_child_map[parent_id].append(code_data)
                except IndexError: continue
        return all_data, parent_child_map

    def _get_distractors(self, item_to_exclude: dict, source_pool: list, count: int, key='code') -> list:
        exclude_value = item_to_exclude[key]
        possible_distractors = [item for item in source_pool if item[key] != exclude_value]
        return random.sample(possible_distractors, min(count, len(possible_distractors)))

    def _generate_benchmark(self, items_pool: list, distractors_pool: list) -> list:
        benchmark_data = []
        if len(distractors_pool) < 4: return []
        for item in items_pool:
            question_text = re.sub(r'\s*\([A-Z0-9.-]+\)$', '', item['description']).strip()
            distractors = self._get_distractors(item, distractors_pool, 3, key='code')
            if len(distractors) < 3: continue
            choices = [item['code']] + [d['code'] for d in distractors]
            random.shuffle(choices)
            benchmark_data.append({'question': question_text, 'choices': choices, 'answer_index': choices.index(item['code'])})
        return benchmark_data

    def _generate_benchmark_reverse(self, items_pool: list, distractors_pool: list) -> list:
        benchmark_data = []
        if len(distractors_pool) < 4: return []
        for item in items_pool:
            question_text = item['code']
            
            # <<< ИСПРАВЛЕНИЕ ЗДЕСЬ >>>
            # Убираем код из описания, чтобы избежать утечки данных.
            # Например, "БОЛЕЗНИ (A00-B99)" -> "БОЛЕЗНИ"
            clean_description = re.sub(r'\s*\([A-Z0-9.-]+\)$', '', item['description']).strip()

            distractors = self._get_distractors(item, distractors_pool, 3, key='description')
            if len(distractors) < 3: continue
            
            # Также очищаем описания дистракторов
            distractor_descriptions = [re.sub(r'\s*\([A-Z0-9.-]+\)$', '', d['description']).strip() for d in distractors]
            
            choices = [clean_description] + distractor_descriptions
            random.shuffle(choices)
            benchmark_data.append({'question': question_text, 'choices': choices, 'answer_index': choices.index(clean_description)})
        return benchmark_data

    # --- Задачи ---
    def generate_task_block_code_by_description(self): return self._generate_benchmark(self.block_codes, self.block_codes)
    def generate_task_level1_code_by_description(self): return self._generate_benchmark(self.level1_codes, self.level1_codes)
    def generate_task_sublevel_code_by_description(self): return self._generate_benchmark(self.sublevel_codes, self.sublevel_codes)
    def generate_task_block_description_by_code(self): return self._generate_benchmark_reverse(self.block_codes, self.block_codes)
    def generate_task_level1_description_by_code(self): return self._generate_benchmark_reverse(self.level1_codes, self.level1_codes)
    def generate_task_sublevel_description_by_code(self): return self._generate_benchmark_reverse(self.sublevel_codes, self.sublevel_codes)

    def save_to_csv(self, data: list, output_path: str):
        header, answer_labels = ['question', 'A', 'B', 'C', 'D', 'answer'], ['A', 'B', 'C', 'D']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in data:
                writer.writerow([item['question']] + item['choices'] + [answer_labels[item['answer_index']]])
        print(f"Бенчмарк ({len(data)} строк) сохранен в: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор бенчмарков по МКБ-10.")
    parser.add_argument('--task-type', type=str, default='all',
                        choices=['all', 'block', 'level1', 'sublevel', 'block_reverse', 'level1_reverse', 'sublevel_reverse'],
                        help="Тип задачи. 'all' - сгенерировать все.")
    parser.add_argument('--csv-file', type=str, default='mkb10.csv', help="Путь к исходному CSV.")
    args = parser.parse_args()
    
    try:
        generator = MKB10BenchmarkGenerator(args.csv_file)
        TASKS_TO_RUN = {
            'block': (generator.generate_task_block_code_by_description, 'mkb10_benchmark_block.csv'),
            'level1': (generator.generate_task_level1_code_by_description, 'mkb10_benchmark_level1.csv'),
            'sublevel': (generator.generate_task_sublevel_code_by_description, 'mkb10_benchmark_sublevel.csv'),
            'block_reverse': (generator.generate_task_block_description_by_code, 'mkb10_benchmark_block_reverse.csv'),
            'level1_reverse': (generator.generate_task_level1_description_by_code, 'mkb10_benchmark_level1_reverse.csv'),
            'sublevel_reverse': (generator.generate_task_sublevel_description_by_code, 'mkb10_benchmark_sublevel_reverse.csv'),
        }
        tasks_to_generate = TASKS_TO_RUN.keys() if args.task_type == 'all' else [args.task_type]
        for task_name in tasks_to_generate:
            if task_name not in TASKS_TO_RUN: continue
            gen_func, filename = TASKS_TO_RUN[task_name]
            print(f"\n--- Генерация задачи: '{task_name}' ---")
            task_data = gen_func()
            if task_data: generator.save_to_csv(task_data, filename)
            else: print(f"Не удалось сгенерировать данные.")
    except (FileNotFoundError, ValueError, KeyError) as e: print(f"Ошибка выполнения: {e}")