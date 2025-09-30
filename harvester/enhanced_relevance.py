"""
Enhanced relevance filtering for NanoChemGPT harvester.
Improves reference quality through multi-layered relevance scoring.
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter

@dataclass
class RelevanceScore:
    """Comprehensive relevance scoring for harvested references."""
    content_score: float = 0.0      # Content-based relevance
    domain_score: float = 0.0       # Domain-specific relevance  
    recency_score: float = 0.0      # Publication recency
    quality_score: float = 0.0      # Journal/venue quality
    entity_score: float = 0.0       # Named entity relevance
    total_score: float = 0.0        # Weighted composite score
    reasons: List[str] = None       # Explanation for scoring

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

# Domain-specific vocabulary for nanochemistry
NANOCHEM_MATERIALS = {
    "nanoparticle", "nanoparticles", "nanotube", "nanotubes", "nanowire", "nanowires",
    "nanocrystal", "nanocrystals", "nanosheet", "nanosheets", "quantum dot", "quantum dots",
    "graphene", "carbon nanotube", "fullerene", "perovskite", "metal oxide", "semiconductor",
    "nanocomposite", "nanostructure", "nanostructures", "thin film", "monolayer", "bilayer"
}

SYNTHESIS_ACTIONS = {
    "synthesis", "synthesize", "synthesized", "prepare", "prepared", "fabricate", "fabricated",
    "deposit", "deposited", "grow", "grown", "form", "formed", "produce", "produced",
    "generate", "generated", "create", "created", "assemble", "assembled"
}

CHARACTERIZATION_TERMS = {
    "characterization", "xrd", "x-ray diffraction", "sem", "tem", "afm", "xps", "ftir",
    "raman", "uv-vis", "photoluminescence", "absorption", "emission", "microscopy"
}

HIGH_IMPACT_JOURNALS = {
    "nature", "science", "nature materials", "nature nanotechnology", "advanced materials",
    "nano letters", "acs nano", "small", "advanced functional materials", "chemistry of materials",
    "journal of materials chemistry", "nanoscale", "chemical reviews", "accounts of chemical research"
}

class EnhancedRelevanceFilter:
    """Enhanced relevance filtering for harvested literature."""
    
    def __init__(self, min_year: int = 2018, quality_threshold: float = 0.4):
        self.min_year = min_year
        self.quality_threshold = quality_threshold
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Materials pattern - broader than current implementation
        materials_pattern = "|".join(re.escape(term) for term in NANOCHEM_MATERIALS)
        self.materials_rx = re.compile(rf"\b({materials_pattern})\b", re.IGNORECASE)
        
        # Synthesis pattern
        synthesis_pattern = "|".join(re.escape(term) for term in SYNTHESIS_ACTIONS)
        self.synthesis_rx = re.compile(rf"\b({synthesis_pattern})\b", re.IGNORECASE)
        
        # Characterization pattern
        char_pattern = "|".join(re.escape(term) for term in CHARACTERIZATION_TERMS)
        self.characterization_rx = re.compile(rf"\b({char_pattern})\b", re.IGNORECASE)
        
        # Methods section indicators (enhanced from current)
        self.methods_rx = re.compile(
            r"\b(materials?\s+and\s+methods?|experimental\s+section|experimental\s+procedure|"
            r"synthesis\s+procedure|preparation\s+method|synthetic\s+route)\b", 
            re.IGNORECASE
        )
    
    def score_content_relevance(self, text: str, title: str = "", abstract: str = "") -> float:
        """Score content relevance based on domain-specific terms."""
        if not text:
            return 0.0
        
        # Combine all text for analysis
        full_text = f"{title} {abstract} {text}".lower()
        
        # Count domain-specific terms
        materials_count = len(self.materials_rx.findall(full_text))
        synthesis_count = len(self.synthesis_rx.findall(full_text))
        char_count = len(self.characterization_rx.findall(full_text))
        methods_sections = len(self.methods_rx.findall(full_text))
        
        # Weighted scoring
        content_score = (
            materials_count * 0.3 +
            synthesis_count * 0.4 +
            char_count * 0.2 +
            methods_sections * 0.1
        )
        
        # Normalize by text length (prevent bias toward longer papers)
        text_length = len(full_text.split())
        if text_length > 0:
            content_score = content_score / math.log(text_length + 1) * 10
        
        return min(1.0, content_score)
    
    def score_domain_specificity(self, journal: str, keywords: List[str] = None) -> float:
        """Score domain specificity based on journal and keywords."""
        if not journal:
            journal_score = 0.0
        else:
            journal_lower = journal.lower()
            # Check for high-impact nanotechnology journals
            if any(hj in journal_lower for hj in HIGH_IMPACT_JOURNALS):
                journal_score = 1.0
            elif any(term in journal_lower for term in ["nano", "material", "chemistry"]):
                journal_score = 0.7
            elif any(term in journal_lower for term in ["physics", "science", "nature"]):
                journal_score = 0.5
            else:
                journal_score = 0.2
        
        # Keywords scoring
        keyword_score = 0.0
        if keywords:
            keyword_text = " ".join(keywords).lower()
            material_matches = len(self.materials_rx.findall(keyword_text))
            keyword_score = min(1.0, material_matches * 0.2)
        
        return (journal_score * 0.7) + (keyword_score * 0.3)
    
    def score_recency(self, year: Optional[int]) -> float:
        """Score based on publication recency."""
        if not year or year < self.min_year:
            return 0.0
        
        current_year = 2025  # Update as needed
        years_old = current_year - year
        
        if years_old <= 1:
            return 1.0
        elif years_old <= 3:
            return 0.8
        elif years_old <= 5:
            return 0.6
        elif years_old <= 7:
            return 0.4
        else:
            return 0.2
    
    def score_entity_relevance(self, entities: List[Dict], query_terms: Set[str] = None) -> float:
        """Score based on extracted named entities."""
        if not entities:
            return 0.0
        
        # Count relevant entity types
        material_entities = [e for e in entities if e.get("label") == "MATERIAL"]
        action_entities = [e for e in entities if e.get("label") == "ACTION"]
        
        entity_score = (
            len(material_entities) * 0.4 +
            len(action_entities) * 0.3 +
            len([e for e in entities if e.get("label") in ["TEMP", "TIME", "AMOUNT"]]) * 0.3
        )
        
        # Bonus for query term matches in entities
        if query_terms:
            entity_texts = [e.get("text", "").lower() for e in entities]
            query_matches = sum(1 for et in entity_texts if any(qt in et for qt in query_terms))
            entity_score += query_matches * 0.2
        
        return min(1.0, entity_score / 10.0)  # Normalize
    
    def calculate_relevance_score(self, 
                                paper: Dict,
                                query_terms: Set[str] = None) -> RelevanceScore:
        """Calculate comprehensive relevance score for a paper."""
        
        # Extract paper information
        text = paper.get("text", "")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        journal = paper.get("journal", "")
        year = paper.get("year") or paper.get("pubYear")
        entities = paper.get("entities", [])
        keywords = paper.get("keywords", [])
        
        # Calculate component scores
        content_score = self.score_content_relevance(text, title, abstract)
        domain_score = self.score_domain_specificity(journal, keywords)
        recency_score = self.score_recency(year)
        entity_score = self.score_entity_relevance(entities, query_terms)
        
        # Quality heuristic based on multiple factors
        quality_score = (
            (1.0 if paper.get("doi") else 0.3) * 0.3 +  # DOI presence
            (1.0 if paper.get("isOpenAccess") else 0.5) * 0.2 +  # Open access
            (min(1.0, len(text.split()) / 5000) if text else 0.0) * 0.3 +  # Content completeness
            (1.0 if journal.lower() in [j.lower() for j in HIGH_IMPACT_JOURNALS] else 0.5) * 0.2
        )
        
        # Weighted total score
        total_score = (
            content_score * 0.35 +
            domain_score * 0.25 +
            recency_score * 0.15 +
            quality_score * 0.15 +
            entity_score * 0.10
        )
        
        # Generate explanation
        reasons = []
        if content_score > 0.6:
            reasons.append("high_content_relevance")
        if domain_score > 0.7:
            reasons.append("domain_specific_venue")
        if recency_score > 0.8:
            reasons.append("recent_publication")
        if entity_score > 0.5:
            reasons.append("relevant_entities_extracted")
        if quality_score > 0.7:
            reasons.append("high_quality_indicators")
        
        return RelevanceScore(
            content_score=content_score,
            domain_score=domain_score,
            recency_score=recency_score,
            quality_score=quality_score,
            entity_score=entity_score,
            total_score=total_score,
            reasons=reasons
        )
    
    def filter_papers(self, papers: List[Dict], 
                     query_terms: Set[str] = None,
                     top_k: int = None) -> List[Tuple[Dict, RelevanceScore]]:
        """Filter and rank papers by relevance."""
        
        scored_papers = []
        for paper in papers:
            score = self.calculate_relevance_score(paper, query_terms)
            if score.total_score >= self.quality_threshold:
                scored_papers.append((paper, score))
        
        # Sort by total score descending
        scored_papers.sort(key=lambda x: x[1].total_score, reverse=True)
        
        if top_k:
            scored_papers = scored_papers[:top_k]
        
        return scored_papers

# Helper function to integrate with existing harvester
def enhance_harvester_relevance(papers: List[Dict], 
                              config: Dict = None) -> List[Dict]:
    """
    Enhance existing harvester with improved relevance filtering.
    
    Args:
        papers: List of paper dictionaries from harvester
        config: Configuration options
    
    Returns:
        Filtered and scored papers with relevance metadata
    """
    config = config or {}
    min_year = config.get("min_year", 2018)
    quality_threshold = config.get("quality_threshold", 0.4)
    top_k = config.get("max_papers", None)
    
    # Extract query terms from config if available
    query_terms = set()
    if "queries" in config:
        for query in config["queries"]:
            query_terms.update(query.lower().split())
    
    filter_engine = EnhancedRelevanceFilter(min_year, quality_threshold)
    scored_papers = filter_engine.filter_papers(papers, query_terms, top_k)
    
    # Add relevance metadata to papers
    enhanced_papers = []
    for paper, score in scored_papers:
        paper["relevance_score"] = score.total_score
        paper["relevance_breakdown"] = {
            "content": score.content_score,
            "domain": score.domain_score, 
            "recency": score.recency_score,
            "quality": score.quality_score,
            "entities": score.entity_score
        }
        paper["relevance_reasons"] = score.reasons
        enhanced_papers.append(paper)
    
    return enhanced_papers