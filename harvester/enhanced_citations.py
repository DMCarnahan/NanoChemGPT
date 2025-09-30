"""
Enhanced reference relevance system for NanoChemGPT.
Improves citation quality and reference ranking based on query context.
"""

import re
import math
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class CitationRelevance:
    """Relevance scoring for citations in responses."""
    query_alignment: float = 0.0     # How well citation aligns with query
    content_quality: float = 0.0     # Quality of cited content
    context_match: float = 0.0       # Match with response context
    authority_score: float = 0.0     # Authority/credibility of source
    recency_bonus: float = 0.0       # Recency weighting
    total_score: float = 0.0         # Composite relevance score
    citation_text: str = ""          # Text where citation appears
    reason: str = ""                 # Primary reason for inclusion/ranking

class EnhancedCitationFilter:
    """Enhanced citation filtering and relevance scoring."""
    
    def __init__(self):
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for citation analysis."""
        # Synthesis-related patterns
        self.synthesis_patterns = re.compile(
            r"\b(synthesiz|prepar|fabricat|deposit|grow|form|produc|generat|creat|assembl)\w*\b",
            re.IGNORECASE
        )
        
        # Characterization patterns
        self.characterization_patterns = re.compile(
            r"\b(xrd|x-ray|sem|tem|afm|xps|ftir|raman|uv-vis|photoluminescence|microscopy)\b",
            re.IGNORECASE
        )
        
        # Material-specific patterns
        self.material_patterns = re.compile(
            r"\b(nanoparticle|nanotube|nanowire|nanocrystal|quantum dot|graphene|perovskite)\w*\b",
            re.IGNORECASE
        )
        
        # Procedural language patterns
        self.procedure_patterns = re.compile(
            r"\b(step|method|procedure|protocol|process|technique|approach)\w*\b",
            re.IGNORECASE
        )
    
    def extract_citation_context(self, text: str, citation_nums: List[int], 
                                window_size: int = 150) -> Dict[int, str]:
        """Extract context around each citation for relevance analysis."""
        contexts = {}
        
        # Find citation positions
        citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
        
        for match in re.finditer(citation_pattern, text):
            cited_nums = [int(n.strip()) for n in match.group(1).split(",")]
            start, end = match.span()
            
            # Extract surrounding context
            context_start = max(0, start - window_size)
            context_end = min(len(text), end + window_size)
            context = text[context_start:context_end].strip()
            
            for num in cited_nums:
                if num in citation_nums:
                    if num in contexts:
                        contexts[num] += " | " + context
                    else:
                        contexts[num] = context
        
        return contexts
    
    def score_query_alignment(self, citation_context: str, query: str, 
                             intent: str = "procedure") -> float:
        """Score how well citation aligns with the original query."""
        if not citation_context or not query:
            return 0.0
        
        context_lower = citation_context.lower()
        query_lower = query.lower()
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        query_terms = {t for t in query_terms if len(t) > 2}  # Filter short words
        
        # Count query term matches in context
        term_matches = sum(1 for term in query_terms if term in context_lower)
        
        # Intent-specific bonuses
        intent_bonus = 0.0
        if intent == "procedure":
            if self.synthesis_patterns.search(context_lower):
                intent_bonus += 0.3
            if self.procedure_patterns.search(context_lower):
                intent_bonus += 0.2
        elif intent == "mechanism":
            if re.search(r'\b(mechanism|pathway|process|reaction)\b', context_lower):
                intent_bonus += 0.3
        elif intent == "comparison":
            if re.search(r'\b(compar|versus|better|superior|advantag)\b', context_lower):
                intent_bonus += 0.3
        
        # Material relevance bonus
        if self.material_patterns.search(context_lower):
            intent_bonus += 0.2
        
        # Normalize score
        base_score = term_matches / max(len(query_terms), 1) if query_terms else 0.0
        total_score = min(1.0, base_score + intent_bonus)
        
        return total_score
    
    def score_content_quality(self, reference: Dict) -> float:
        """Score the quality of the reference content."""
        quality_score = 0.0
        
        # Journal/venue quality
        journal = reference.get("journal", "").lower()
        if any(high_impact in journal for high_impact in [
            "nature", "science", "advanced materials", "nano letters", "acs nano"
        ]):
            quality_score += 0.4
        elif any(domain_term in journal for domain_term in [
            "nano", "material", "chemistry", "physics"
        ]):
            quality_score += 0.2
        
        # DOI presence (indicates formal publication)
        if reference.get("doi"):
            quality_score += 0.2
        
        # Open access availability
        if reference.get("isOpenAccess") or reference.get("pdf_url"):
            quality_score += 0.1
        
        # Content completeness
        text_length = len(reference.get("text", ""))
        if text_length > 5000:  # Substantial content
            quality_score += 0.2
        elif text_length > 1000:
            quality_score += 0.1
        
        # Author count (collaborative work often higher quality)
        authors = reference.get("authors", [])
        if len(authors) >= 3:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def score_context_match(self, citation_context: str, response_text: str) -> float:
        """Score how well the citation fits within the response context."""
        if not citation_context or not response_text:
            return 0.0
        
        context_lower = citation_context.lower()
        response_lower = response_text.lower()
        
        # Extract key terms from both
        context_terms = set(re.findall(r'\b\w+\b', context_lower))
        response_terms = set(re.findall(r'\b\w+\b', response_lower))
        
        # Filter out common words
        stopwords = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        context_terms = {t for t in context_terms if len(t) > 2 and t not in stopwords}
        response_terms = {t for t in response_terms if len(t) > 2 and t not in stopwords}
        
        if not context_terms or not response_terms:
            return 0.0
        
        # Calculate overlap
        overlap = len(context_terms & response_terms)
        union = len(context_terms | response_terms)
        
        jaccard_score = overlap / union if union > 0 else 0.0
        
        # Bonus for technical term matches
        tech_terms = {"synthesis", "characterization", "nanoparticle", "temperature", "concentration"}
        tech_overlap = len(tech_terms & context_terms & response_terms)
        tech_bonus = tech_overlap * 0.1
        
        return min(1.0, jaccard_score + tech_bonus)
    
    def score_authority(self, reference: Dict, citation_count: int = 0) -> float:
        """Score the authority/credibility of the reference."""
        authority_score = 0.0
        
        # Journal impact (simplified heuristic)
        journal = reference.get("journal", "").lower()
        if "nature" in journal or "science" in journal:
            authority_score += 0.5
        elif any(term in journal for term in ["advanced", "nano", "materials", "chemistry"]):
            authority_score += 0.3
        elif "journal" in journal:
            authority_score += 0.2
        
        # Publication year (recent work often more authoritative)
        year = reference.get("year") or reference.get("pubYear")
        if year:
            years_old = 2025 - int(year)  # Update current year as needed
            if years_old <= 2:
                authority_score += 0.3
            elif years_old <= 5:
                authority_score += 0.2
            elif years_old <= 10:
                authority_score += 0.1
        
        # Citation frequency in response (multiple citations suggest importance)
        if citation_count > 1:
            authority_score += min(0.2, citation_count * 0.05)
        
        return min(1.0, authority_score)
    
    def rank_citations(self, references: List[Dict], response_text: str, 
                      query: str, intent: str = "procedure") -> List[Tuple[int, CitationRelevance]]:
        """Rank citations by relevance to query and response context."""
        
        # Extract citation numbers from response
        citation_nums = []
        for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', response_text):
            nums = [int(n.strip()) for n in match.group(1).split(",")]
            citation_nums.extend(nums)
        
        citation_counter = Counter(citation_nums)
        
        # Extract contexts for each citation
        contexts = self.extract_citation_context(response_text, citation_nums)
        
        rankings = []
        
        for i, ref in enumerate(references, 1):
            if i not in citation_nums:
                continue  # Skip unused citations
            
            citation_context = contexts.get(i, "")
            citation_count = citation_counter.get(i, 0)
            
            # Calculate component scores
            query_alignment = self.score_query_alignment(citation_context, query, intent)
            content_quality = self.score_content_quality(ref)
            context_match = self.score_context_match(citation_context, response_text)
            authority_score = self.score_authority(ref, citation_count)
            
            # Recency bonus
            year = ref.get("year") or ref.get("pubYear")
            recency_bonus = 0.0
            if year:
                years_old = 2025 - int(year)
                if years_old <= 1:
                    recency_bonus = 0.2
                elif years_old <= 3:
                    recency_bonus = 0.1
            
            # Weighted total score
            total_score = (
                query_alignment * 0.3 +
                content_quality * 0.25 +
                context_match * 0.25 +
                authority_score * 0.15 +
                recency_bonus * 0.05
            )
            
            # Determine primary reason for relevance
            reason = "general"
            if query_alignment > 0.7:
                reason = "high_query_alignment"
            elif content_quality > 0.8:
                reason = "high_quality_source"
            elif context_match > 0.6:
                reason = "strong_context_match"
            elif authority_score > 0.7:
                reason = "authoritative_source"
            
            relevance = CitationRelevance(
                query_alignment=query_alignment,
                content_quality=content_quality,
                context_match=context_match,
                authority_score=authority_score,
                recency_bonus=recency_bonus,
                total_score=total_score,
                citation_text=citation_context,
                reason=reason
            )
            
            rankings.append((i, relevance))
        
        # Sort by total score descending
        rankings.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return rankings
    
    def filter_low_relevance_citations(self, response_text: str, references: List[Dict],
                                     query: str, intent: str = "procedure",
                                     min_score: float = 0.3) -> Tuple[str, List[Dict]]:
        """Remove low-relevance citations and reorder references."""
        
        rankings = self.rank_citations(references, response_text, query, intent)
        
        # Filter citations below threshold
        relevant_citations = [(num, rel) for num, rel in rankings if rel.total_score >= min_score]
        
        if not relevant_citations:
            # Keep at least one citation if all are below threshold
            relevant_citations = rankings[:1] if rankings else []
        
        # Create mapping from old to new citation numbers
        citation_map = {}
        filtered_references = []
        
        for new_num, (old_num, relevance) in enumerate(relevant_citations, 1):
            citation_map[old_num] = new_num
            filtered_references.append(references[old_num - 1])  # 0-indexed
        
        # Update citation numbers in response text
        def replace_citation(match):
            cited_nums = [int(n.strip()) for n in match.group(1).split(",")]
            new_nums = [citation_map[n] for n in cited_nums if n in citation_map]
            if new_nums:
                return f"[{', '.join(map(str, sorted(new_nums)))}]"
            else:
                return ""  # Remove citation entirely
        
        filtered_response = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace_citation, response_text)
        
        # Clean up any double spaces left by removed citations
        filtered_response = re.sub(r'\s+', ' ', filtered_response).strip()
        
        return filtered_response, filtered_references

# Helper function to integrate with existing ref_utils
def enhance_citation_relevance(response_text: str, references: List[Dict], 
                             query: str, intent: str = "procedure") -> Tuple[str, List[Dict]]:
    """
    Enhance citation relevance in generated responses.
    
    Args:
        response_text: Generated response with citations
        references: List of reference dictionaries
        query: Original user query
        intent: Query intent (procedure/comparison/mechanism/definition)
    
    Returns:
        Tuple of (filtered_response, filtered_references)
    """
    if not references or not response_text:
        return response_text, references
    
    filter_engine = EnhancedCitationFilter()
    return filter_engine.filter_low_relevance_citations(
        response_text, references, query, intent, min_score=0.25
    )