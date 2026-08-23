# Reviewer analyses — verbatim record

Every substantive review, claim, and suggestion from the project's expert
reviewer, recovered from the session transcript and preserved here as the
primary source the probe suite and decision fixes are distilled from.
Severity language is the reviewer's own and is load-bearing: "obvious
mistake" claims come with an explicit epistemic contract (see 2026-07-29).

---

## 2026-07-27 (message 8)

I don’t know how useful giving specific is in terms of helping you train a stronger ai but it’s still making very elemental mistakes
Also, while training on point margins is obviously good and very useful in terms of getting the model to play decently, all that matters in the end is whether you win or lose. So once the model can handle it, if it has for example a choice to win 5% by making some play but losing by a larger margin in the other 95% of the time, it should of course pick this over losing 100% of the time even if this would massively increase the average amount it lost by
I don’t know how much elaborating helps you train a better model, but it’s still making very elementary mistakes.

Also, ultimately winning or losing is all that matters, of course a reward system that takes into account point differentials is very important to train the model to play well, once the model is solid enough, it should be trained to focus on winning matches. (a match being 3 rounds) . For clarity, as an example, if the model can make a play that increases its chance of winning, while massively increasing the expected value of average points it loses by, ideally it should make the play that increases its chance of winning.

---

## 2026-07-28 (message 9)

Can you do some testing on how rollout search compares to policy and in which situations? to try to improve when to use which or above which policy threshhold to evaluate the options with rollout etc

---

## 2026-07-28 (message 10)

I’m not sure what you mean with ‘doomed opens’ or ‘wasted wagers’, but if it simply means expeditions/wagers with negative points this seems like a misguided metric to track, because with optimal play they are obviously both non-zero and you don’t have a reference point for what the numbers should be under optimal play, and they are currently low enough that they shouldn’t necessarily be  lower.

The current viewer is still showing the old game (would it be possible for you to check the rollout search value for ply 5 play W4 instead of disc Yx? they seem fairly close to me but Yx has 100% policy prior so I can’t see the score)

Once I have the new model in the viewer I’ll see if I can spot mistakes

---

## 2026-07-28 (message 11)

Huh I actually would have guessed for ply1 discarding Yx would NOT be the correct play, and play Gx and disc Y2 both seem better, but I’m only human so this is not a strong claim. Could you check if rollout search is still superior to policy at the early stages of rounds (rounds not matches)

---

## 2026-07-28 (message 16)

But can you check ply 5 please? It's still playing disc Yx when play W4 seems better

---

## 2026-07-28 (message 17)

It seems like (early game at least) rollout search is way too noisy at 96 worlds to be helpful at all. at ply 1 it thinks disc Yx is by far best when it clearly isn't, at ply 4 it think's drawing yellow is best when that's very bad for sure. I know this is just variance, but it seems to be that you either need to use more worlds or do something else to make it useful. 

Also, at ply 97, P1 has 77% belief that P2 has Wx even though P2 drew Wx from the discard pile. Afaict usually the certain hands don't show up so this seems like a bug. 

Also, the rollout search numbers makes it seems like it's only considering point EV unless i'm misunderstanding how it works.

---

## 2026-07-28 (message 18)

Why are we only doing 300 pairs for big decision like that? Keep trying to improve the ai please

---

## 2026-07-28 (message 19)

Would it be possible for you to implement ruling out decision that are STRICTLY inferior? as an example, at ply 25, when discarding, discarding B4 is strictly dominating all other discards because there is no chance that it is useful to either player, while disc Rx (while only 0.1% chosen, the fact that it's chosen at all is bad) could help the opponent. This should make the sims run quicker because less decisions need to be evaluated. This should ONLY be done for decisions that are STRICTLY inferior to others (unless I'm missing some reason here why it wouldn't be strictly inferior)

---

## 2026-07-28 (message 20)

Huh, yeah your pile burial thing is a good point. I doubt that bait discards are a meaningful strategic thing in perfect play. Please make sure the pile burial thing isn't impactful enough for us not to make this change. Ply 32 again it tries to play W7 when it's not good.

---

## 2026-07-28 (message 21)

ply 39, drawing from the yellow pile seems completely awful, considering that P2 just discarded a yellow, and W10 is still possibly in the deck with P2 having 2 white wagers, and we don't have enough cardsleftto play to needto stall yet

---

## 2026-07-28 (message 22)

No, I'm sorry, but you are wrong. 22% chance of W10 being in the deck is very very significant when white has 2 white wagers down. 

"Meanwhile drawing the dead-ish Y6 preserves one deck card, marginally stretching the window P0 needs to unload G7-G8-G10 (+75 with the double wager)."
I already adressed this in my previous comment, we have 4 playable cards left with 9 cards in deck left. It is literally impossible for stalling to have any value yet.

---

## 2026-07-28 (message 23)

For ply 72 could you try to figure out why rollout search thinks play W4 is better than play B8? The only reason not to play B8 is information denial so if P1 has B9 they don't have an easy play and might need to discard or play something suboptimal. I'd like to know if that's the actual reason or it's just rollout search being wrong.

---

## 2026-07-28 (message 24)

Ok, we should be trying to make the AI better so it doesn't keep having to keep making bad plays to protect itself from even worse plays in the future. is it better for me to collect the mistakes into one comment and give them at one time or write them one message at a time?

---

## 2026-07-29 (message 26)

Ply1: not saying at all this is a mistake, could you run some analysis on what makes disc W2 not better than play Y2? To me W2 seems like the best play, and interestingly on ply3 disc W2 is suddenly the best play according to search even though not much changed to suddenly make it better than play Bx or play G3 compared to ply1

Ply 10: play G5 is SO much better than play R5 (and i would agree with this) according to rollout, but policy says play R5 70% of the time, so this is an error we have to work on wrt policy

Ply 12: I don't see much reason to play Wx over G6, and many reasons to play G6 instead, so G6 should clearly be better i think

ply 13: play G4 should clearly be better than disc Yx, see literally no reason for disc Yx to be better

ply 19: play B7 should certainly be better than play Y10? B7 loses out on max 10 points in case of B5 draw, while Y10 loses out on potential Y789, PLUS playing B7 lets us play B8 and B9 without drawbacks so we're sure we won't have to make any potentially costly plays for a while

ply 27: policy wants to disc R2 2% of time which is completely nonsensical

ply 36: this is a very important flaw the model still has. It currently has 6 playable cards. There is never a reason to start stalling by drawing a pointless discard at this point. It just helps your opponent. It must simply be protecting itself against future mistakes.

ply 37: no reason at all to disc W6 over play G9, the fact that rollout thinks disc W6 can be better indicates a clear flaw

ply 38: again, no reason to stall yet

ply 43: completely, completely awful rollout decision to draw from Y discard. very very obvious blunder and flaw.

ply 46: obvious blunder to play W10, there are no playable cards we can possible get besides W9,so we should just draw from deck to see if we get W9 until we run out of time. Also obvious blunder to take R3 discard instead of drawing from deck.

Ply 54: play Y3 seems like a valid option or the best play, could you check that please, it's not in the top 4 so it's not considered. Also, rollout thinks disc Y4 is better than disc Y3 which is obvious nonsense.

ply 58: Slightly surprised that play Gx is better than disc B2 but not making any claims about this.

ply 68: taking from Blue discard pile seems bad but rollout thinks disc Y8->B is better than disc Y8-> deck, but the play ends up being play G7 -> deck, but there 's no reason why this play would choose a different draw. Anyway, ->B seems bad

ply 74: search thinking disc Y8-> is better than Bx->B seems like nonsense.

ply 93: complete, complete, complete nonsense to disc R2 instead of playing W9. Search thinking disc R2 is so much better is a very very very clear flaw.

ply 111:very clear flaw. absolutely no reason to not play G10, disc Y5 when opponent has Yx down instead of G10 when we already have G9 played is very clear nonsense.

Ply 112: We should obviously play R4->Y instead

ply 113: We should obviously play G10, I really don't understand what is causing this behaviour, playing Bx instead is complete nonsense

ply 114: again, we should be drawing from Y, not deck

ply 115: should play G10 again

ply 116: again, should be drawing from Y

ply 118: MAYBE G8 and draw from Y is better but because the policy is only considering Y7->Y as a draw from Y play, which is obviously nonsense, we don't get to see

ply 122: search thinks disc Rx is better than play Y10 which is nonsense.

ply 125: COMPLETE BLUNDER and showing a very obvious flaw in search. policy wants to play W5 which is very very clearly the correct play. search does disc G6 instead, while opponent has Gx down and playing W5 has no downsides. clear flaw

ply 126: we are not considering drawing G6 for some reason, same for ply 128


ply 130: completely horrible. We should very very very obviously play G8 over disc Gx. Play G8 is obviously the best play if we're not drawing the G6 from disc and it's not even close.

ply 134: should play W8

ply 143: obvious error, since we don't even stall to be able to play our reds in the next plies. 

PAY attention to how egregious I've said the mistakes are.
These are a lot of errors so we really need to take good care to take everything into account

---

## 2026-07-29 (message 27)

We really can't patch every obvious mistake like this. you need to use my input to improve POLICY training too.

When I say something is an obvious mistake, please don't just run a test and dismiss what I said based on the results. I will only say obvious mistake if it's obvious. You should assume the search is either wrong or the EV of the mistake is simply too small to catch.

For the mistakes i pointed out as clear mistakes, that you want to dismiss, you would need to point out the flaw in my logic. For move 130 for example.

---

## 2026-07-29 (message 28)

Ok, please improve the AI.

Also, why are you using deterministic playouts? for analysis obviously the deck should be reshuffled every new world, otherwise a bad play that is the best play GIVEN the current deck order would be seen as the best play, which is obvious nonsense

---

## 2026-07-29 (message 33)

I would like an artifact where I have a convenient UI to plays against the AI. After the game I would like all my moves to be analyzed and all close or wrong decision to be in a list to review

---

## 2026-07-29 (message 35)

Is there a way to add a complete solver for the last x plies of the rounds that can run fast enough? should be doable? For example, I dnno if you can see the last game i played against the AI, but on the very last card of a deck, it's policy was to 94% of the the time make a play that simply loses 10 points, and it's the last card so this is a very simple play. it wanted to make the correct play 0% of the time. incase you can see history, it was play W10 v play Y10 when one of them had a wager card down.

---

## 2026-07-29 (message 38)

What i meant with make the tree way smaller is that it should make training way more effective, because it means that each position gets 'seen' 5 times as often as when you don't make them equivalent

---

## 2026-07-31 (message 40)

I am going to make your best ai play againt another models best ai. I will be creating an environment for the matches be simulated. Please list ALL the files in the github that are necessary for your model to play at full strength

---

## 2026-08-06 (message 43)

There are just too many horrible plays in here for me to review, I really have a hard time believing the AI has actually been getting that much better, is it possible for you to test this version against one of the old versions? please keep trying to improve the ai

---

## 2026-08-17 (message 45)

ply 3 also it makes a weird play because of search, I haven't even checked the rest of the game. Why did we start using search so early now in this way?

---

## 2026-08-18 (message 46)

Ply 19:
completely insanely bad play.

also whydid search override policy here when the chosen option doesn't even score well in search??

---

## 2026-08-21 (message 47)

Ply 2 the policy prior lists play Yx->deck twice. Have we still not implemented all simple symmetry simplifications like that yet?

---

## 2026-08-22 (message 48)

ply 7 seems bad to play Y4 when opponent has 2 Yx down, play W8 seems better

ply 8 disc Wx seems better  than play R2

ply 13: wtf? absolutely horrendous discarding Yx when opponent has 2 Yx down and you're not blocking any yellow. play B6 or R7 are obviously better

ply23: disc W2 seems clearly best

ply 28: why is the search considering drawing from the W pile?

ply 31: disc W2 or play Y6 are obviously better 

ply 33: play Y6 seems clearly better than play B8

ply 35: again play Y6 seems better

ply 38: stalling too early, obvious flaw 

ply 43: also stalling when not needed

ply 47: literally no playable cards left, just nonsense to stall, but since it's known that opponent also can't have any playable cards in this instance it doesn't matter

ply 59: play Gx seems kinda sonnsense here, B6 seems best

ply 89: horrible stall, complete nonsense, we only have 3 playable cards

ply 91: again complete nonsense stall, just helping opponent, just a very obvious sign something is wrong with the ai

ply 93: same again


Not reviewing 3rd game, this seems like enough

---

## 2026-08-22 (message 49)

Is it possible to test the current strongest AI on ALL the plies I have brought to your intention in the past with either my comment on what the best play is or plies I wanted analysed , and give the analysis to me and also try to use it to still improve the current AI?

---

## 2026-08-23 (message 50)

I gave way way way more clear analyses of turns than the ones that show up in your analysis, are they not available anymore because of context limit or chat condensation? are they nowhere in the github?

---

## 2026-08-23 (mid-turn)

Also, for ply 38 you said the stall is good, but we only have 5 cards to play
with 8 cards in deck so it's too early to stall

---

## 2026-08-23 (mid-turn)

"Wager-gift prune — discarding a wager the opponent can still play is removed
from the candidate set outright (...)"

Also, this is not a good fix, please undo this. It can absolutely be best at
times to discard a wager the opponent can still play. Please try to avoid
patchy fixes like this unless for cases that are very very clear, we should
just be improving the ai so it doesn't make bad plays
