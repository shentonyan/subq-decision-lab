"""One call (~200 tokens, ~$0.00001) to check the key, credits and response shape."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subq_lab.client import SubQClient, choice, noul

c = SubQClient()
r = c.decide(
    {"mechanism": "sealed-bid second-price auction, one item", "bids": {"A": 10, "B": 7, "C": 12}},
    {
        "winner": choice("Which bidder wins?", {"A": None, "B": None, "C": None}),
        "truthful": noul("Is truthful bidding weakly dominant in this format?"),
    },
)
print(json.dumps(r, indent=2))
print(c.summary())
