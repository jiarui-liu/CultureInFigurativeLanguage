# Final reviewer-style audit

## Central claim and novelty

The paper asks one question: what a language model learns from a culture's idioms. Its answer separates encountered meaning, figurative transfer, and cultural symbolism. IdiomAtlas supplies the missing representation that connects an expression's figurative meaning to its literal imagery and entities across four languages.

## Evidence chain

- Every training claim uses token-matched Random-CPT as the primary control.
- Idiom-Text versus Idiom+Meaning separates running context from explicit dictionary supervision.
- Encountered meanings improve in Arabic, Hindi, and Chinese at 9B and reproduce at 2B.
- Arabic provides held-out figurative transfer on expressions almost entirely absent from the knowledge base.
- The symbolism result is supported by both item-weighted and language-weighted aggregation, a decision-margin measure, and an independently fixed 376-item consensus subset.
- The Hindi symbolism-statement intervention holds probe entities out of the appended statements, making its gain a transfer result rather than answer recall.
- The Arabic unseen-idiom decrease from appended meanings remains in the main paper because it is necessary to interpret the mechanism.

## Claim calibration

The paper does not claim universal gains on factual culture knowledge. It attributes learning to the symbolic and evaluative layer measured by idiom and symbolism tasks. Per-language symbolism estimates remain visible, including Arabic, beside the pooled result. Paired intervals describe item uncertainty, and the one-seed limitation is explicit.

## Presentation and reproducibility

The abstract, introduction, result figures, and conclusion use the same three-level framing. Main-text result grids were replaced by generated figures, while exact values and robustness checks remain in the appendix. Figure scripts, statistical scripts, cached audit decisions, a claim ledger, and regeneration commands are stored in the code repository.
