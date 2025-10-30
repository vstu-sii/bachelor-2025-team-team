import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class LLMEvaluator:
    """Оценка качества генерации моделей LLM"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def calculate_bleu_score(self, generated, reference, n=4):
        """Упрощённый расчёт BLEU score (precision + brevity penalty)"""
        def get_ngrams(text, n):
            words = text.split()
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        generated_ngrams = get_ngrams(generated, n)
        reference_ngrams = get_ngrams(reference, n)
        
        if not generated_ngrams or not reference_ngrams:
            return 0.0
        
        matches = len(set(generated_ngrams) & set(reference_ngrams))
        precision = matches / len(generated_ngrams)
        
        # Brevity penalty (штраф за короткие ответы)
        bp = 1.0 if len(generated.split()) > len(reference.split()) else np.exp(
            1 - len(reference.split()) / max(len(generated.split()), 1)
        )
        return bp * precision
    
    def calculate_rouge_score(self, generated, reference):
        """Упрощённый ROUGE-L score (на основе LCS)"""
        def longest_common_subsequence(words1, words2):
            m, n = len(words1), len(words2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if words1[i-1] == words2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        if not generated or not reference:
            return 0.0
        
        words1, words2 = generated.split(), reference.split()
        lcs = longest_common_subsequence(words1, words2)
        
        recall = lcs / len(words2) if words2 else 0
        precision = lcs / len(words1) if words1 else 0
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)
    
    def calculate_semantic_similarity(self, text1, text2):
        """Семантическое сходство на основе TF-IDF + cosine similarity"""
        if not text1 or not text2:
            return 0.0
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(sim[0][0])
        except Exception:
            return 0.0
    
    def detect_hallucinations(self, generated_text, source_context):
        """Оценка "галлюцинаций" — предложений, которых нет в контексте"""
        if not generated_text or not source_context:
            return 0.0
        
        generated_sentences = [s.strip() for s in re.split(r'[.!?]', generated_text) if s.strip()]
        context_sentences = [s.strip() for s in re.split(r'[.!?]', source_context) if s.strip()]
        
        if not generated_sentences:
            return 0.0
        
        hallucinated = 0
        for sentence in generated_sentences:
            if not any(
                self.calculate_semantic_similarity(sentence, ctx) > 0.6
                for ctx in context_sentences
            ):
                hallucinated += 1
        
        return hallucinated / len(generated_sentences)
    
    def evaluate_response_quality(self, generated_data, reference_data, context=None):
        """Комплексная оценка качества ответа модели"""
        metrics = {}
        
        # Преобразуем данные в текст
        generated_text = json.dumps(generated_data, ensure_ascii=False) if isinstance(generated_data, (dict, list)) else str(generated_data)
        reference_text = json.dumps(reference_data, ensure_ascii=False) if isinstance(reference_data, (dict, list)) else str(reference_data)
        
        # Метрики качества
        metrics['bleu'] = self.calculate_bleu_score(generated_text, reference_text)
        metrics['rouge'] = self.calculate_rouge_score(generated_text, reference_text)
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(generated_text, reference_text)
        metrics['hallucination_rate'] = self.detect_hallucinations(generated_text, context) if context else 0.0
        
        # Проверка на структурную корректность
        metrics['structural_validity'] = 1.0 if isinstance(generated_data, dict) else 0.0
        
        # Веса для интегральной метрики
        weights = {
            'bleu': 0.2,
            'rouge': 0.3,
            'semantic_similarity': 0.3,
            'structural_validity': 0.2
        }
        
        metrics['overall_score'] = sum(metrics[m] * w for m, w in weights.items())
        return metrics


class PerformanceMetrics:
    """Метрики производительности"""
    
    @staticmethod
    def calculate_latency_metrics(latencies):
        """Агрегация статистики по времени отклика"""
        if not latencies:
            return {
                'mean_latency': 0,
                'median_latency': 0,
                'p95_latency': 0,
                'min_latency': 0,
                'max_latency': 0,
                'std_latency': 0
            }
        
        return {
            'mean_latency': float(np.mean(latencies)),
            'median_latency': float(np.median(latencies)),
            'p95_latency': float(np.percentile(latencies, 95)),
            'min_latency': float(np.min(latencies)),
            'max_latency': float(np.max(latencies)),
            'std_latency': float(np.std(latencies))
        }
    
    @staticmethod
    def calculate_throughput(total_requests, total_time):
        """Пропускная способность (req/sec)"""
        return total_requests / total_time if total_time > 0 else 0
    
    @staticmethod
    def calculate_success_rate(successful_requests, total_requests):
        """Процент успешных ответов"""
        return successful_requests / total_requests if total_requests > 0 else 0
