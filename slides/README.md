# Pitch deck

The anomality pitch deck: 12 slides, made as a claude.ai Slides artifact
(https://claude.ai/artifact/Tq6Dh5mXrHvzc6VFMr5o4S, private until shared from its Share menu).
Open it there to present, or download it as PowerPoint or PDF from the deck page.

This folder is the deck's source, so it lives with the code:

- `deck.json`: title, slide order, sections and fonts (Space Grotesk, IBM Plex Sans).
- `slides/<id>.html`: one 1920x1080 slide each, with the speaker notes in its `<aside>`.

| # | Slide | |
|---|---|---|
| 1 | `cover` | anomality |
| 2 | `problem` | risk models break in a crash: 102%, 11.7σ, 93% |
| 3 | `original` | the original RG-ResMoE architecture (redrawn from Ye & Borde) |
| 4 | `failure` | in a crash the original gate stops working |
| 5 | `ours` | our architecture: log inputs + the decision layer |
| 6 | `layer` | the decision layer: one question a day |
| 7 | `system` | the full system: offline training, live app, baseline |
| 8 | `forecast` | forecast quality vs. the original and the simple estimate |
| 9 | `decision` | the mood rule's backtest |
| 10 | `product` | the weekly report |
| 11 | `honest` | what we don't claim, and what we do |
| 12 | `demo` | the 45 s app demo video |

Images and the video are stored in the artifact and referenced as `/_blob/<id>`. The same files in this repo:

| Reference | File |
|---|---|
| `/_blob/c884dd3d4d3b0795ec33438f0ec95f4d` | `docs/images/diagram_failure_and_fix.png` |
| `/_blob/54e9f8d2223516eb18d8d87d109a34b4` | `docs/images/product_decision.png` |
| `/_blob/32fd3a1b9025566163573f0e73a92401` | `docs/images/product_anomaly_report.png` |
| `/_blob/cb9571a30937be0016fd02924fb3bccc` | `demo/renders/anomality-demo.mp4` |
| `/_blob/3ff83e62e6aee5af15f6f096dd7ec553` | `slides/demo-still.jpg` (the video's poster frame) |

The cover still has placeholders to fill in: `[Hackathon name] · [Team names]`.
