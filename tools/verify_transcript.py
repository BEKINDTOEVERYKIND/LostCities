#!/usr/bin/env python3
"""Replay a showgame transcript against an independent implementation of the rules.

Nothing here shares code with the C engine: the point is that a transcript the
engine produced still checks out when the rules are written a second time.
"""
import re
import sys

SUITS = "YBWGR"
SUIT_BY_NAME = {"Yellow": "Y", "Blue": "B", "White": "W", "Green": "G", "Red": "R"}


def parse(path):
    hands, moves, expeditions, final = {}, [], {}, {}
    player = None
    for line in open(path):
        m = re.match(r"Player (\d) hand:\s+(.*)", line)
        if m:
            hands[int(m.group(1))] = m.group(2).split()
            continue
        m = re.match(r"\s*(\d+)\s+P(\d)\s+(play|discard)\s+(\S+)\s+(\S+) \((\S+)\)", line)
        if m:
            moves.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4),
                          m.group(5), m.group(6)))
            continue
        m = re.match(r"\s*Player (\d)\s*$", line)
        if m:
            player = int(m.group(1))
            continue
        m = re.match(r"\s+(Yellow|Blue|White|Green|Red)\s+(-|[YBWGRx0-9 ]+?)\s+(?:\(|-$)", line)
        if m and player:
            cards = [] if m.group(2).strip() == "-" else m.group(2).split()
            expeditions[(player, SUIT_BY_NAME[m.group(1)])] = cards
            continue
        m = re.match(r"final score: Player 1 ([+-]\d+), Player 2 ([+-]\d+)", line)
        if m:
            final = {1: int(m.group(1)), 2: int(m.group(2))}
    return hands, moves, expeditions, final


def value(card):
    return 0 if card[1] == "x" else int(card[1:])


def main(path):
    hands, moves, expeditions, final = parse(path)
    errs = []

    # --- the deck must be exactly the 60 real cards -------------------------
    full = [s + "x" for s in SUITS for _ in range(3)] + [s + str(v) for s in SUITS for v in range(2, 11)]
    assert len(full) == 60

    hand = {1: list(hands[1]), 2: list(hands[2])}
    piles = {s: [] for s in SUITS}
    played = {(p, s): [] for p in (1, 2) for s in SUITS}
    seen = list(hand[1]) + list(hand[2])          # cards revealed so far
    deck_draws = 0
    expect_turn = 1

    for ply, p, act, card, src, drawn in moves:
        if p != expect_turn:
            errs.append(f"ply {ply}: turn order broken (expected P{expect_turn})")
        expect_turn = 3 - p
        suit = card[0]

        if card not in hand[p]:
            errs.append(f"ply {ply}: P{p} used {card}, not in hand {sorted(hand[p])}")
        else:
            hand[p].remove(card)

        if act == "play":
            nums = [c for c in played[(p, suit)] if c[1] != "x"]
            if card[1] == "x":
                if nums:
                    errs.append(f"ply {ply}: wager {card} after number cards {nums}")
            elif nums and value(card) <= max(value(c) for c in nums):
                errs.append(f"ply {ply}: {card} not ascending over {nums}")
            played[(p, suit)].append(card)
        else:
            piles[suit].append(card)

        if src == "deck":
            deck_draws += 1
            seen.append(drawn)
        else:
            ps = SUIT_BY_NAME[src]
            if act == "discard" and ps == suit:
                errs.append(f"ply {ply}: took back the card just discarded to {src}")
            if not piles[ps]:
                errs.append(f"ply {ply}: drew from empty {src} pile")
            elif piles[ps][-1] != drawn:
                errs.append(f"ply {ply}: drew {drawn} from {src}, top was {piles[ps][-1]}")
                piles[ps].pop()
            else:
                piles[ps].pop()
        hand[p].append(drawn)

        if len(hand[p]) != 8:
            errs.append(f"ply {ply}: P{p} hand size {len(hand[p])}")

    # --- deck accounting ---------------------------------------------------
    if deck_draws != 44:
        errs.append(f"deck draws {deck_draws}, expected 44")
    if moves[-1][4] != "deck":
        errs.append("last ply did not draw from the deck, so the game could not have ended")

    # --- every card exists exactly as often as the deck allows -------------
    # The three wagers of a suit share a name, so this is a multiset test: the
    # 16 dealt cards plus the 44 deck draws must be exactly the 60 card deck.
    if sorted(seen) != sorted(full):
        from collections import Counter
        got, want = Counter(seen), Counter(full)
        for card in sorted(set(full) | set(seen)):
            if got[card] != want[card]:
                errs.append(f"{card}: revealed {got[card]} times, deck holds {want[card]}")

    # --- transcript's expedition listing must match the replay -------------
    for key, cards in played.items():
        listed = expeditions.get(key, [])
        if sorted(cards) != sorted(listed):
            errs.append(f"P{key[0]} {key[1]}: replay {sorted(cards)} vs listed {sorted(listed)}")

    # --- scoring -----------------------------------------------------------
    totals = {}
    for p in (1, 2):
        tot = 0
        for s in SUITS:
            cards = played[(p, s)]
            if not cards:
                continue
            wag = sum(1 for c in cards if c[1] == "x")
            tot += (sum(value(c) for c in cards) - 20) * (1 + wag) + (20 if len(cards) >= 8 else 0)
        totals[p] = tot
    if totals != final:
        errs.append(f"recomputed score {totals} vs printed {final}")

    left = sum(len(v) for v in played.values()) + sum(len(v) for v in piles.values()) + 16
    if left != 60:
        errs.append(f"cards accounted for: {left}, expected 60")

    print(f"replayed {len(moves)} plies, {deck_draws} deck draws, "
          f"{len(moves) - deck_draws} pile draws")
    print(f"independent score: P1 {totals[1]:+d}, P2 {totals[2]:+d} "
          f"(transcript says P1 {final[1]:+d}, P2 {final[2]:+d})")
    if errs:
        print(f"\n{len(errs)} PROBLEMS:")
        for e in errs:
            print("  " + e)
        return 1
    print("no rule violations, no accounting errors, scoring matches")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/game.txt"))
