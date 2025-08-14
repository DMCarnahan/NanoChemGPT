from miner import BasicMiner

miner = BasicMiner()

def extract_procedure(text: str) -> dict:
    """
    Returns {"operations":[{op_type, materials, params, sentence, offsets...}], "expanded":[...micro_steps...]}
    """
    ops = miner.extract(text)
    expanded = miner.expand(ops)
    return {"operations": ops, "expanded": expanded}