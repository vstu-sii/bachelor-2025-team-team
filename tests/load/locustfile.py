from locust import HttpUser, SequentialTaskSet, task, between, events
from locust.exception import StopUser
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESUME_DIR = Path(r"C:\Users\67181\OneDrive\Dokumenty\bachelor-2025-team-team\data\test_resumes")

# ===== НАСТРОЙКИ RETRY =====
MAX_RETRIES = 3
BACKOFF_BASE = 2  # 2^attempt секунд

# ===== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ =====
def post_with_retry(client, url, *, files=None, json_data=None, name=None):
    for attempt in range(1, MAX_RETRIES + 1):
        with client.post(
            url,
            files=files,
            json=json_data,
            name=name,
            catch_response=True
        ) as response:

            if response.status_code == 200:
                return response

            if response.status_code == 429 and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                response.failure(f"429 → retry in {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            response.failure(response.text)
            return None

    return None


class HRPipelineTasks(SequentialTaskSet):

    def on_start(self):
        # Каждый пользователь получает ПОЛНЫЙ список файлов
        self.resume_files = [
            f for f in RESUME_DIR.iterdir()
            if f.suffix.lower() in (".doc", ".docx", ".txt")
        ]

        self.current_index = 0

        vacancy_file = RESUME_DIR.parent / "vacancy.json"
        with open(vacancy_file, "r", encoding="utf-8") as f:
            self.vacancy_json = json.load(f)

        self.analysis_result = None

    @task
    def pipeline(self):
        # Все файлы обработаны → пользователь завершён
        if self.current_index >= len(self.resume_files):
            logging.info(f"User {id(self.user)} finished all resumes")
            raise StopUser()


        resume_file = self.resume_files[self.current_index]
        logging.info(
            f"User {id(self.user)} processing "
            f"{resume_file.name} ({self.current_index + 1}/{len(self.resume_files)})"
        )

        # ===== ШАГ 1: extract =====
        with open(resume_file, "rb") as f:
            response = post_with_retry(
                self.client,
                "/extract-resume",
                files={"file": (resume_file.name, f)},
                name="extract_resume"
            )

        if not response:
            self.interrupt()
            return

        self.analysis_result = response.json().get("analysis_result")
        if not self.analysis_result:
            self.interrupt()
            return

        # ===== ШАГ 2: match =====
        response = post_with_retry(
            self.client,
            "/match-vacancy",
            json_data={
                "analysis_result": self.analysis_result,
                "vacancy_data": self.vacancy_json
            },
            name="match_vacancy"
        )

        if not response:
            self.interrupt()
            return

        # ===== ШАГ 3: generate =====
        response = post_with_retry(
            self.client,
            "/generate-interview-plan",
            json_data={
                "analysis_result": self.analysis_result,
                "vacancy_requirements": self.vacancy_json
            },
            name="generate_questions"
        )

        if not response:
            self.interrupt()
            return

        # Переходим к следующему файлу
        self.current_index += 1

@events.quitting.add_listener
def generate_text_report(environment, **kwargs):
    stats = environment.stats

    report_lines = []

    report_lines.append("LOCUST LOAD TEST REPORT")
    report_lines.append("=" * 22)
    report_lines.append("")
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(
        f"Active users: {environment.runner.user_count if environment.runner else 0}"
    )
    report_lines.append("")
    report_lines.append("--- REQUEST STATISTICS ---")
    report_lines.append("")

    for (req_type, name), entry in stats.entries.items():
        report_lines.append(f"{req_type} {name}")
        report_lines.append(f"  Requests: {entry.num_requests}")
        report_lines.append(f"  Failures: {entry.num_failures}")
        report_lines.append(f"  Avg response time: {entry.avg_response_time:.2f} ms")
        report_lines.append(
            f"  P95 response time: {entry.get_response_time_percentile(0.95):.2f} ms"
        )
        report_lines.append(f"  RPS: {entry.current_rps:.2f}")
        report_lines.append("")

    report_lines.append("--- PIPELINE STATISTICS ---")
    report_lines.append("")

    pipeline_entry = stats.get("PIPELINE", "full_pipeline")
    if pipeline_entry:
        report_lines.append("Full pipeline:")
        report_lines.append(f"  Completed: {pipeline_entry.num_requests}")
        report_lines.append(f"  Failed: {pipeline_entry.num_failures}")
        report_lines.append(
            f"  Avg total time: {pipeline_entry.avg_response_time:.2f} ms"
        )
        report_lines.append(
            f"  P95 total time: {pipeline_entry.get_response_time_percentile(0.95):.2f} ms"
        )

    report_path = Path("locust_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 Text report saved to {report_path.resolve()}")
        
    
    


class HRApiUser(HttpUser):
    wait_time = between(0.5, 1)
    tasks = [HRPipelineTasks]
