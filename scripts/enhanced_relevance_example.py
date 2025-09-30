"""
Example integration of enhanced relevance filtering in NanoChemGPT harvester.
Shows how to use the new relevance and citation systems.
"""

import yaml
from pathlib import Path
from harvester.enhanced_relevance import enhance_harvester_relevance
from harvester.enhanced_citations import enhance_citation_relevance

def example_enhanced_harvesting():
    """Example of using enhanced relevance filtering during harvesting."""
    
    # Load existing config
    config_path = Path("harvester/config.yaml")
    config = yaml.safe_load(config_path.read_text())
    
    # Enhance config with relevance settings
    config.update({
        "min_year": 2020,              # Only recent papers
        "quality_threshold": 0.5,      # Higher quality threshold
        "max_papers": 100,             # Limit total papers
        "enable_enhanced_relevance": True
    })
    
    # Simulate papers from harvester (normally from harvest_one_record)
    sample_papers = [
        {
            "title": "Synthesis of Gold Nanoparticles via Chemical Reduction",
            "abstract": "We describe a novel method for synthesizing gold nanoparticles...",
            "journal": "Nano Letters",
            "year": 2023,
            "doi": "10.1021/nl2023456",
            "isOpenAccess": True,
            "text": "Gold nanoparticles were synthesized using sodium borohydride reduction...",
            "entities": [
                {"label": "MATERIAL", "text": "gold nanoparticles"},
                {"label": "ACTION", "text": "synthesized"},
                {"label": "MATERIAL", "text": "sodium borohydride"}
            ]
        },
        {
            "title": "Review of Nanomaterial Safety",
            "journal": "Environmental Science",
            "year": 2019,
            "text": "This review covers safety aspects of various nanomaterials..."
        }
    ]
    
    # Apply enhanced relevance filtering
    enhanced_papers = enhance_harvester_relevance(sample_papers, config)
    
    print(f"Original papers: {len(sample_papers)}")
    print(f"After relevance filtering: {len(enhanced_papers)}")
    
    for paper in enhanced_papers:
        print(f"Title: {paper['title']}")
        print(f"Relevance Score: {paper['relevance_score']:.3f}")
        print(f"Breakdown: {paper['relevance_breakdown']}")
        print(f"Reasons: {paper['relevance_reasons']}")
        print("---")

def example_enhanced_citations():
    """Example of using enhanced citation filtering in responses."""
    
    # Simulate LLM response with citations
    response_text = """
    Gold nanoparticles can be synthesized through chemical reduction methods [1]. 
    The most common approach involves using sodium borohydride as a reducing agent [1,2].
    Characterization typically involves XRD and TEM analysis [3].
    Safety considerations are important when working with nanomaterials [2].
    """
    
    # Simulate reference list
    references = [
        {
            "title": "Synthesis of Gold Nanoparticles via Chemical Reduction",
            "journal": "Nano Letters",
            "year": 2023,
            "doi": "10.1021/nl2023456",
            "text": "Gold nanoparticles were synthesized using sodium borohydride reduction at room temperature..."
        },
        {
            "title": "Review of Nanomaterial Safety",
            "journal": "Environmental Science", 
            "year": 2019,
            "text": "This review covers safety aspects of various nanomaterials in industrial applications..."
        },
        {
            "title": "XRD Characterization of Nanoparticles",
            "journal": "Characterization Methods",
            "year": 2022,
            "text": "X-ray diffraction provides valuable information about crystal structure of nanoparticles..."
        }
    ]
    
    # Apply enhanced citation filtering
    query = "How to synthesize gold nanoparticles?"
    intent = "procedure"
    
    filtered_response, filtered_refs = enhance_citation_relevance(
        response_text, references, query, intent
    )
    
    print("Original response:")
    print(response_text)
    print(f"Original references: {len(references)}")
    print()
    
    print("Enhanced response:")
    print(filtered_response)
    print(f"Filtered references: {len(filtered_refs)}")
    print()
    
    print("Reference relevance analysis:")
    from harvester.enhanced_citations import EnhancedCitationFilter
    filter_engine = EnhancedCitationFilter()
    rankings = filter_engine.rank_citations(references, response_text, query, intent)
    
    for cite_num, relevance in rankings:
        print(f"Citation [{cite_num}]: {relevance.total_score:.3f}")
        print(f"  Query alignment: {relevance.query_alignment:.3f}")
        print(f"  Content quality: {relevance.content_quality:.3f}")
        print(f"  Context match: {relevance.context_match:.3f}")
        print(f"  Authority: {relevance.authority_score:.3f}")
        print(f"  Reason: {relevance.reason}")
        print()

if __name__ == "__main__":
    print("=== Enhanced Harvester Relevance Example ===")
    example_enhanced_harvesting()
    print()
    
    print("=== Enhanced Citation Relevance Example ===")
    example_enhanced_citations()