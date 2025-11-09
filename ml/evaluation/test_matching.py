import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import glob
from typing import Dict, List, Any

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import Gemma3Text

load_dotenv()


class Evaluator:
    """Класс для оценки правильности и структуры сгенерированных файлов"""
    
    def __init__(self):
        self.results_dir = "ml/evaluation/evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.report_data = []
    
    def evaluate_baseline_integration(self, pipeline) -> Dict[str, Any]:
        """
        Оценивает интеграцию с baseline.py с реальными файлами
        
        Args:
            pipeline: Объект пайплаина для анализа резюме
            
        Returns:
            Dict с результатами оценки
        """
        print("🚀 Запуск оценки baseline интеграции...")
        
        evaluation_result = {
            "test_name": "baseline_integration",
            "total_files": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "file_results": [],
            "errors": []
        }
        
        # Поиск тестовых файлов
        test_files = self._find_test_files()
        evaluation_result["total_files"] = len(test_files)
        
        if not test_files:
            evaluation_result["errors"].append("Тестовые файлы не найдены")
            print("❌ Тестовые файлы не найдены.")
            return evaluation_result
        
        print(f"✅ Найдено {len(test_files)} тестовых файлов")
        
        # Обработка каждого файла
        print(f"📊 Обрабатываем {len(test_files)} файлов...")
        
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
            
            file_result = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "status": "unknown",
                "analysis_result": None,
                "error": None
            }
            
            try:
                # Анализируем резюме
                analysis_result = pipeline.analyze_resume(file_path)
                
                if "error" in analysis_result:
                    file_result["status"] = "failed"
                    file_result["error"] = analysis_result["error"]
                    evaluation_result["failed_analyses"] += 1
                    print(f"❌ Ошибка анализа: {analysis_result['error']}")
                else:
                    file_result["status"] = "success"
                    file_result["analysis_result"] = analysis_result
                    evaluation_result["successful_analyses"] += 1
                    print("✅ Анализ завершен успешно!")
                    
                    # Сохраняем анализ в JSON файл
                    self._save_analysis_result(analysis_result, file_path)
                    
            except Exception as e:
                file_result["status"] = "error"
                file_result["error"] = str(e)
                evaluation_result["failed_analyses"] += 1
                print(f"❌ Критическая ошибка: {e}")
            
            evaluation_result["file_results"].append(file_result)
        
        # Формируем итоговую статистику
        success_rate = (evaluation_result["successful_analyses"] / evaluation_result["total_files"] * 100 
                       if evaluation_result["total_files"] > 0 else 0)
        
        evaluation_result["success_rate"] = round(success_rate, 2)
        
        print(f"\n🎯 Оценка завершена! Успешно обработано: {evaluation_result['successful_analyses']}/{evaluation_result['total_files']} файлов")
        print(f"📊 Успешность: {evaluation_result['success_rate']}%")
        
        # Сохраняем результаты оценки
        self._save_evaluation_result(evaluation_result, "baseline_integration")
        self.report_data.append(evaluation_result)
        
        return evaluation_result

    def evaluate_vacancy_matching(self, pipeline) -> Dict[str, Any]:
        """
        Оценивает модуль сопоставления вакансий с резюме
        
        Args:
            pipeline: Объект пайплаина для анализа резюме
            
        Returns:
            Dict с результатами оценки
        """
        print("🚀 Запуск оценки модуля сопоставления вакансий с резюме...")
        
        evaluation_result = {
            "test_name": "vacancy_matching",
            "total_candidates": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "vacancy_data": None,
            "matching_results": [],
            "ranking_results": [],
            "errors": []
        }
        
        # Шаг 1: Подготавливаем данные вакансии
        print("\n1. 📋 Подготавливаем данные вакансии...")
        vacancy_data = self._create_test_vacancy()
        evaluation_result["vacancy_data"] = vacancy_data
        
        # Сохраняем данные вакансии
        vacancy_file = os.path.join(self.results_dir, "vacancy_data.json")
        with open(vacancy_file, 'w', encoding='utf-8') as f:
            json.dump(vacancy_data, f, ensure_ascii=False, indent=2)
        print("💾 Сохранены данные вакансии")
        
        # Шаг 2: Загружаем и анализируем резюме
        print("\n2. 📄 Загружаем и анализируем резюме...")
        analyzed_candidates = pipeline._load_and_analyze_resumes(["ml/evaluation/test_analysis_results/resume_ranking_test"])
        
        if not analyzed_candidates:
            evaluation_result["errors"].append("Нет подходящих резюме для анализа")
            print("❌ Нет подходящих резюме для анализа")
            return evaluation_result
        
        evaluation_result["total_candidates"] = len(analyzed_candidates)
        print(f"✅ Загружено {len(analyzed_candidates)} резюме")
        
        # Шаг 3: Сопоставляем каждого кандидата с вакансией
        print("\n3. 🎯 Сопоставляем кандидатов с вакансией...")
        matching_results = []
        
        for i, candidate in enumerate(analyzed_candidates, 1):
            print(f"\n--- Кандидат {i}/{len(analyzed_candidates)}: {candidate['filename']} ---")
            
            candidate_result = {
                "candidate_id": candidate["filename"],
                "resume_analysis": candidate["resume_analysis"],
                "status": "unknown",
                "matching_result": None,
                "overall_score": 0,
                "is_suitable": False,
                "error": None
            }
            
            try:
                # Сопоставляем с вакансией
                matching_result = pipeline.match_vacancy_with_resume(
                    candidate["resume_analysis"], 
                    vacancy_data
                )
                
                if "error" in matching_result:
                    candidate_result["status"] = "failed"
                    candidate_result["error"] = matching_result["error"]
                    evaluation_result["failed_matches"] += 1
                    print(f"❌ Ошибка сопоставления: {matching_result['error']}")
                else:
                    candidate_result["status"] = "success"
                    candidate_result["matching_result"] = matching_result
                    candidate_result["overall_score"] = matching_result.get("matching_results", {}).get("overall_score", 0)
                    candidate_result["is_suitable"] = matching_result.get("matching_results", {}).get("is_suitable", False)
                    
                    evaluation_result["successful_matches"] += 1
                    print(f"✅ Оценка: {candidate_result['overall_score']}/100")
                    
                    # Сохраняем индивидуальный результат
                    self._save_individual_matching_result(candidate_result, i)
                    
            except Exception as e:
                candidate_result["status"] = "error"
                candidate_result["error"] = str(e)
                evaluation_result["failed_matches"] += 1
                print(f"❌ Критическая ошибка: {e}")
            
            matching_results.append(candidate_result)
            evaluation_result["matching_results"].append(candidate_result)
        
        # Шаг 4: Ранжируем кандидатов
        print("\n4. 📊 Ранжируем кандидатов...")
        ranked_candidates = self._rank_candidates(matching_results)
        evaluation_result["ranking_results"] = ranked_candidates
        
        # Шаг 5: Сохраняем финальные результаты
        print("\n5. 💾 Сохраняем финальные результаты...")
        self._save_final_matching_results(evaluation_result)
        
        # Формируем статистику
        success_rate = (evaluation_result["successful_matches"] / evaluation_result["total_candidates"] * 100 
                       if evaluation_result["total_candidates"] > 0 else 0)
        
        evaluation_result["success_rate"] = round(success_rate, 2)
        evaluation_result["suitable_candidates"] = len([c for c in matching_results if c["is_suitable"]])
        
        print(f"\n🎯 Оценка завершена!")
        print(f"📊 Успешно сопоставлено: {evaluation_result['successful_matches']}/{evaluation_result['total_candidates']}")
        print(f"📈 Подходящих кандидатов: {evaluation_result['suitable_candidates']}")
        print(f"📊 Успешность: {evaluation_result['success_rate']}%")
        
        # Сохраняем результаты оценки
        self._save_evaluation_result(evaluation_result, "vacancy_matching")
        self.report_data.append(evaluation_result)
        
        return evaluation_result

    def _create_test_vacancy(self) -> Dict[str, Any]:
        """
        Создает тестовые данные вакансии
        
        Returns:
            Dict с данными вакансии
        """
        vacancy_data = {
            "job_title": "Middle Backend Developer (Java/Kotlin)",
            "education": "Высшее техническое образование (компьютерные науки, программная инженерия, информационные технологии)",
            "work_experience": 3,
            "desired_salary": 180000,
            "work_schedule": "Полный день",
            "work_format": "Гибридный формат (2-3 дня в офисе, остальное время удаленно)",
            "additional_requirements": "Опыт работы с Java 8+, Kotlin, Spring Boot, микросервисной архитектурой, PostgreSQL, Redis, Docker, Kubernetes, знание английского языка на уровне чтения технической документации"
        }
        
        print("📋 Тестовая вакансия:")
        for key, value in vacancy_data.items():
            print(f"   • {key}: {value}")
        
        return vacancy_data

    def _save_individual_matching_result(self, candidate_data: Dict, index: int) -> None:
        """
        Сохраняет индивидуальный результат сопоставления
        
        Args:
            candidate_data: Данные кандидата
            index: Индекс кандидата
        """
        candidate_dir = os.path.join(self.results_dir, f"candidate_{index:02d}_{candidate_data['candidate_id']}")
        os.makedirs(candidate_dir, exist_ok=True)
        
        # Сохраняем результат сопоставления
        matching_file = os.path.join(candidate_dir, "matching_result.json")
        try:
            with open(matching_file, 'w', encoding='utf-8') as f:
                json.dump(candidate_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Сохранен результат для кандидата {index}")
        except Exception as e:
            print(f"❌ Ошибка сохранения результата кандидата: {e}")

    def _rank_candidates(self, matching_results: List[Dict]) -> List[Dict]:
        """
        Ранжирует кандидатов по оценке
        
        Args:
            matching_results: Результаты сопоставления
            
        Returns:
            List отсортированных кандидатов
        """
        # Фильтруем успешные результаты
        successful_results = [r for r in matching_results if r["status"] == "success"]
        
        # Сортируем по убыванию оценки
        ranked_candidates = sorted(
            successful_results, 
            key=lambda x: x["overall_score"], 
            reverse=True
        )
        
        print(f"📊 Ранжировано {len(ranked_candidates)} кандидатов")
        
        # Выводим топ-3 кандидатов
        if ranked_candidates:
            print("\n🏆 Топ-3 кандидатов:")
            for i, candidate in enumerate(ranked_candidates[:3], 1):
                print(f"   {i}. {candidate['candidate_id']} - {candidate['overall_score']}/100")
        
        return ranked_candidates

    def _save_final_matching_results(self, evaluation_result: Dict) -> None:
        """
        Сохраняет финальные результаты сопоставления
        
        Args:
            evaluation_result: Результаты оценки
        """
        final_results = {
            "vacancy_data": evaluation_result["vacancy_data"],
            "summary": {
                "total_candidates": evaluation_result["total_candidates"],
                "successful_matches": evaluation_result["successful_matches"],
                "failed_matches": evaluation_result["failed_matches"],
                "suitable_candidates": evaluation_result["suitable_candidates"],
                "success_rate": evaluation_result["success_rate"]
            },
            "ranking_results": evaluation_result["ranking_results"]
        }
        
        final_file = os.path.join(self.results_dir, "final_matching_results.json")
        try:
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            print("💾 Сохранены финальные результаты сопоставления")
        except Exception as e:
            print(f"❌ Ошибка сохранения финальных результатов: {e}")

    def _find_test_files(self) -> List[str]:
        """
        Ищет тестовые файлы в различных папках
        
        Returns:
            List[str]: Список путей к тестовым файлам
        """
        test_folders = [
            "data/test_resumes"
        ]
        
        test_files = []
        for folder in test_folders:
            if os.path.exists(folder):
                # Ищем все поддерживаемые форматы
                patterns = [
                    f"{folder}/*.pdf", 
                    f"{folder}/*.docx", 
                    f"{folder}/*.doc", 
                    f"{folder}/*.txt"
                ]
                for pattern in patterns:
                    found_files = glob.glob(pattern)
                    test_files.extend(found_files)
        
        return test_files
    
    def _save_analysis_result(self, analysis_result: Dict, file_path: str) -> None:
        """
        Сохраняет результат анализа в JSON файл
        
        Args:
            analysis_result: Результат анализа
            file_path: Путь к исходному файлу
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        json_filename = f"resume_analysis_{file_name}.json"
        json_path = os.path.join(self.results_dir, json_filename)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            print(f"💾 Сохранен JSON: {json_filename}")
        except Exception as e:
            print(f"❌ Ошибка сохранения JSON: {e}")
    
    def _save_evaluation_result(self, evaluation_result: Dict, test_name: str) -> None:
        """
        Сохраняет результаты оценки в JSON файл
        
        Args:
            evaluation_result: Результаты оценки
            test_name: Название теста
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{test_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты оценки сохранены: {filename}")
        except Exception as e:
            print(f"❌ Ошибка сохранения результатов оценки: {e}")
    
    def generate_report(self) -> str:
        """
        Генерирует текстовый отчет на основе всех проведенных оценок
        
        Returns:
            str: Текстовый отчет
        """
        if not self.report_data:
            return "📊 ОТЧЕТ ОЦЕНКИ СИСТЕМЫ\n\nФункции оценки не выполнены. Запустите методы evaluate_* для генерации отчета."
        
        report = "📊 ОТЧЕТ ОЦЕНКИ СИСТЕМЫ\n\n"
        report += "=" * 50 + "\n\n"
        
        for eval_data in self.report_data:
            test_name = eval_data.get("test_name", "Unknown")
            report += f"🔹 {test_name.upper()}\n"
            report += "-" * 30 + "\n"
            
            if test_name == "baseline_integration":
                report += self._format_baseline_report(eval_data)
            elif test_name == "vacancy_matching":
                report += self._format_matching_report(eval_data)
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report
    
    def _format_baseline_report(self, data: Dict) -> str:
        """Форматирует отчет для baseline интеграции"""
        report = f"📁 Всего файлов: {data['total_files']}\n"
        report += f"✅ Успешно обработано: {data['successful_analyses']}\n"
        report += f"❌ Ошибок: {data['failed_analyses']}\n"
        report += f"📊 Успешность: {data['success_rate']}%\n"
        return report
    
    def _format_matching_report(self, data: Dict) -> str:
        """Форматирует отчет для сопоставления вакансий"""
        report = f"📋 Вакансия: {data['vacancy_data']['job_title']}\n"
        report += f"👥 Всего кандидатов: {data['total_candidates']}\n"
        report += f"✅ Успешно сопоставлено: {data['successful_matches']}\n"
        report += f"🎯 Подходящих кандидатов: {data['suitable_candidates']}\n"
        report += f"📊 Успешность: {data['success_rate']}%\n"
        
        if data['ranking_results']:
            report += f"🏆 Топ-1 кандидат: {data['ranking_results'][0]['candidate_id']} ({data['ranking_results'][0]['overall_score']}/100)\n"
        
        return report

# Пример использования
if __name__ == "__main__":
    evaluator = Evaluator()
    # Для тестирования нужно будет передать pipeline
    # results1 = evaluator.evaluate_baseline_integration(pipeline)
    # results2 = evaluator.evaluate_vacancy_matching(pipeline)
    # print(evaluator.generate_report())