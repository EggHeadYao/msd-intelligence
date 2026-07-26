            Recommendation(track_id, score, rank, frozenset({"audio"}), {"cos_audio": score})
            for rank, (track_id, score) in enumerate(filtered[:k], start=1)
        ]
        audit = ColdAudioAudit(len(raw), len(filtered), self.candidate_limit - len(filtered))
        return recommendations, audit
