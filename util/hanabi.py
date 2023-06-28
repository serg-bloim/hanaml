from core.replay import Card


def generate_all_cards():
    for c in "rygbw":
        for n, cnt in enumerate([3, 2, 2, 2, 1], start=1):
            yield from [Card.from_str(f"{c}{n}")] * cnt
