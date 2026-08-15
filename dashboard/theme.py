"""
Design tokens and CSS for the dashboard, glassmorphism direction: a
deep indigo-to-violet gradient background, frosted translucent glass
cards with soft glow, and two glowing accent hues reserved specifically
for the isolated and connected experimental conditions, everywhere in
the app, not used decoratively for anything else. Color still encodes
real meaning here, only the visual language around it changed.
"""

BG_DEEP = "#0B1130"
GLASS_BG = "rgba(255, 255, 255, 0.055)"
GLASS_BORDER = "rgba(255, 255, 255, 0.14)"
TEXT_PRIMARY = "#EAF0FF"
TEXT_MUTED = "#9BA7C9"

ISOLATED = "#22D3EE"    # cyan glow, reserved meaning: the "isolated" condition
CONNECTED = "#C084FC"   # violet glow, reserved meaning: the "connected" condition
ALERT = "#FF6584"
GRADIENT_A = "#4318FF"
GRADIENT_B = "#0B1130"

# used only for series with no isolated/connected meaning (e.g. rule-based agents A/B/C)
NEUTRAL_PALETTE = ["#7DD3FC", ISOLATED, CONNECTED]


def palette_for(columns) -> list:
    """
    Assigns colors to a list of chart series names. Uses the reserved
    condition colors when a name signals isolated/connected, otherwise
    cycles a neutral palette. Keeps color meaningful where the content
    has real structure, and plain where it doesn't.
    """
    colors = []
    for i, col in enumerate(columns):
        key = str(col).lower()
        if "isolated" in key:
            colors.append(ISOLATED)
        elif "connected" in key:
            colors.append(CONNECTED)
        else:
            colors.append(NEUTRAL_PALETTE[i % len(NEUTRAL_PALETTE)])
    return colors


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 15% 10%, rgba(67, 24, 255, 0.35), transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(34, 211, 238, 0.18), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(192, 132, 252, 0.16), transparent 45%),
        {BG_DEEP};
    background-attachment: fixed;
}}

.glass-card {{
    background: {GLASS_BG};
    border: 1px solid {GLASS_BORDER};
    border-radius: 20px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    padding: 2rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}}

.hero .eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {ISOLATED};
    margin-bottom: 0.7rem;
}}
.hero h1 {{
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.3;
    color: {TEXT_PRIMARY};
    margin: 0 0 0.7rem 0;
}}
.hero p {{
    color: {TEXT_MUTED};
    font-size: 1rem;
    max-width: 680px;
    margin-bottom: 1.5rem;
}}

.ticker-row {{
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}}
.chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid {GLASS_BORDER};
    border-radius: 999px;
    padding: 0.5rem 1rem 0.5rem 0.7rem;
    color: {TEXT_PRIMARY};
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}}
.chip .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {ISOLATED};
    box-shadow: 0 0 8px {ISOLATED};
    display: inline-block;
}}
.chip.teal .dot {{
    background: {CONNECTED};
    box-shadow: 0 0 8px {CONNECTED};
}}
.chip b {{ color: {ISOLATED}; font-weight: 600; }}
.chip.teal b {{ color: {CONNECTED}; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.4rem;
}}
[data-testid="stTabs"] button {{
    background: {GLASS_BG};
    border: 1px solid {GLASS_BORDER};
    border-radius: 999px;
    padding: 0.3rem 0.4rem;
}}
[data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 500;
    font-size: 0.92rem;
    color: {TEXT_MUTED};
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    background: linear-gradient(135deg, {GRADIENT_A}, {ISOLATED});
    border: none;
}}
[data-testid="stTabs"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_PRIMARY};
    font-weight: 600;
}}

[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT_PRIMARY};
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED};
}}

.section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ISOLATED};
    margin-bottom: 0.4rem;
}}

.stButton button {{
    background: linear-gradient(135deg, {GRADIENT_A}, {ISOLATED});
    color: {TEXT_PRIMARY};
    border: none;
    border-radius: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(67, 24, 255, 0.45);
}}
.stButton button:hover {{
    box-shadow: 0 6px 26px rgba(34, 211, 238, 0.5);
}}

[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
}}
</style>
"""
