import app_utils.pdf_utils as pu
import app_utils.text_chunks as tc
import app_utils.fact_extractor as fe
import app_utils.rendering as rd

print('OK: helper modules imported')
print('pdf normalize sample:', pu.normalize_pdf_text('a·b·c'))
print('pick method sample:', tc.pick_method_paragraph('This is intro\n\nHeat to 100 °C in a water bath.'))
print('facts sample:', fe.extract_facts_from_text('Heat 100 mL of 0.1 m FeSO4 in a water bath at 50 °C.'))
