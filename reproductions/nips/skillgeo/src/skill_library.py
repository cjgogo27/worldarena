from dataclasses import asdict

from typing import Any



import numpy as np

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer



from .skill_parser import Skill





class SkillLibrary:

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:

        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

        self.skills: list[Skill] = []

        self._bm25: BM25Okapi | None = None

        self._tokenized_corpus: list[list[str]] = []

        self._embeddings: np.ndarray | None = None



    @staticmethod

    def _tokenize(text: str) -> list[str]:

        return [t for t in text.lower().replace("\n", " ").split(" ") if t]



    def add_skills(self, skills: list[Skill]) -> None:

        if not skills:

            return

        self.skills.extend(skills)

        self._tokenized_corpus = [self._tokenize(skill.skill_text) for skill in self.skills]

        self._bm25 = BM25Okapi(self._tokenized_corpus)

        self._embeddings = self.embedding_model.encode(

            [skill.skill_text for skill in self.skills],

            normalize_embeddings=True,

            convert_to_numpy=True,

            show_progress_bar=False,

        )



    def retrieve(

        self,

        query_text: str,

        top_k: int = 5,

        alpha: float = 0.5,

        min_score: float = 0.0,

        deduplicate_region: bool = True,

    ) -> list[dict[str, Any]]:

        if not self.skills:

            return []

        if self._bm25 is None or self._embeddings is None:

            raise RuntimeError("Skill library not indexed")



        bm25_scores = np.array(self._bm25.get_scores(self._tokenize(query_text)), dtype=np.float32)

        query_emb = self.embedding_model.encode(

            [query_text],

            normalize_embeddings=True,

            convert_to_numpy=True,

            show_progress_bar=False,

        )[0]

        sem_scores = self._embeddings @ query_emb



        bm25_norm = self._normalize_scores(bm25_scores)

        sem_norm = self._normalize_scores(sem_scores)

        combined = alpha * sem_norm + (1.0 - alpha) * bm25_norm



        candidate_idxs = np.argsort(-combined)

        results: list[dict[str, Any]] = []

        seen_regions: set[str] = set()



        for idx in candidate_idxs:

            score = float(combined[int(idx)])

            if score < min_score and len(results) >= 2:

                break

            skill = self.skills[int(idx)]



            if deduplicate_region and len(results) >= 3:

                if skill.region_hint in seen_regions and skill.region_hint != "unknown":

                    continue



            payload = asdict(skill)

            payload["score"] = score

            payload["bm25_score"] = float(bm25_scores[int(idx)])

            payload["semantic_score"] = float(sem_scores[int(idx)])

            results.append(payload)

            if skill.region_hint != "unknown":

                seen_regions.add(skill.region_hint)



            if len(results) >= top_k:

                break



        return results



    def retrieve_multi(

        self,

        queries: list[tuple[str, float]],

        top_k: int = 8,

        alpha: float = 0.5,

        min_score: float = 0.0,

        deduplicate_region: bool = False,

    ) -> list[dict[str, Any]]:

        if not queries:

            return []



        accum: dict[tuple[str, str, int], dict[str, Any]] = {}

        total_w = sum(max(0.0, w) for _, w in queries) or 1.0



        for query_text, weight in queries:

            if not query_text.strip() or weight <= 0:

                continue

            partial = self.retrieve(

                query_text=query_text,

                top_k=max(top_k * 3, 12),

                alpha=alpha,

                min_score=min_score,

                deduplicate_region=deduplicate_region,

            )

            scaled_w = weight / total_w

            for p in partial:

                key = (str(p.get("source_game_id", "")), str(p.get("skill_text", "")), int(p.get("source_round", 0)))

                if key not in accum:

                    payload = dict(p)

                    payload["score"] = 0.0

                    accum[key] = payload

                accum[key]["score"] += float(p.get("score", 0.0)) * scaled_w



        merged = sorted(accum.values(), key=lambda x: -float(x.get("score", 0.0)))

        return merged[:top_k]



    @staticmethod

    def _normalize_scores(arr: np.ndarray) -> np.ndarray:

        if arr.size == 0:

            return arr

        mn = float(arr.min())

        mx = float(arr.max())

        if abs(mx - mn) < 1e-8:

            return np.zeros_like(arr, dtype=np.float32)

        return (arr - mn) / (mx - mn)
