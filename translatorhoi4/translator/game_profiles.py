"""Game-specific translation profiles for improved localization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GameProfile:
    """Configuration for a specific game's translation requirements."""
    game_id: str
    name: str
    genre: str
    style: str
    key_terms: List[str]
    notes: str
    tone: str = "formal"  # formal, informal, creative, technical


# Dictionary of supported game profiles
GAME_PROFILES: Dict[str, GameProfile] = {
    "hoi4": GameProfile(
        game_id="hoi4",
        name="Hearts of Iron IV",
        genre="WW2 Grand Strategy",
        style="Military, political, historical",
        key_terms=[
            "division", "brigade", "battalion", "regiment", "corps", "army", "army group",
            "front", "ideology", "focus tree", "national focus", "research slot",
            "political power", "stability", "war support", "command power",
            "equipment", "manpower", "factory", "dockyard", "military factory",
            "civilian factory", "synthetic refinery", "officer corps",
            "division template", "combat width", "organization", "hardness",
            "soft attack", "hard attack", "defense", "breakthrough",
            "entrenchment", "recon", "support company", "battalion",
            "democratic", "fascist", "communist", "non-aligned", "neutrality",
            "puppet", "annex", "liberate", "faction", "volunteer",
        ],
        notes=(
            "Use historically accurate terminology. Military terms must be precise and consistent. "
            "Country names, leader names, and historical events should match official translations. "
            "Keep formal military and political language. Avoid colloquialisms. "
            "WW2-era terminology should be period-accurate. "
            "Focus tree names and national spirit names are often creative - preserve their flavor."
        ),
        tone="formal",
    ),
    "ck3": GameProfile(
        game_id="ck3",
        name="Crusader Kings III",
        genre="Medieval Dynasty Simulator",
        style="Medieval, feudal, dynastic, religious",
        key_terms=[
            "duchy", "county", "kingdom", "empire", "barony", "castle", "city", "temple",
            "liege", "vassal", "succession", "feudal", "clan", "tribal", "elective",
            "prestige", "piety", "gold", "renown", "legitimacy", "tyranny",
            "opinion", "realm", "council", "chancellor", "steward", "marshal",
            "spymaster", "chaplain", "courtier", "knight", "men-at-arms",
            "culture", "faith", "religion", "doctrine", "tenet", "virtue", "sin",
            "dynasty", "house", "bloodline", "heritage", "tradition",
            "title", "de jure", "de facto", "claim", "pressed claim", "weak claim",
        ],
        notes=(
            "Medieval terminology is critical. Keep all titles formal and period-appropriate. "
            "Religious terms should match the faith being described. "
            "Dynasty and family terms are very important - use consistent translation. "
            "Cultural names and traditions often have historical basis - research if unsure. "
            "Preserve archaic flavor where appropriate (e.g., 'thou', 'hath' equivalents). "
            "Avoid modern political terms in medieval context."
        ),
        tone="formal",
    ),
    "eu4": GameProfile(
        game_id="eu4",
        name="Europa Universalis IV",
        genre="Early Modern Strategy (1444-1821)",
        style="Colonial, diplomatic, trade-focused, early modern",
        key_terms=[
            "province", "trade", "colonize", "mission", "idea group", "national idea",
            "monarch power", "adm", "dip", "mil", "adm power", "dip power", "mil power",
            "prestige", "legitimacy", "republican tradition", "devotion", "horde unity",
            "mandate of heaven", "imperial authority", "papal influence", "karma",
            "trade power", "trade value", "steer trade", "embargo", "light ship",
            "core", "claim", "overextension", "aggressive expansion", "coalition",
            "estate", "nobility", "clergy", "burghers", "peasants", "cossacks", "tribes",
            "diplomatic reputation", "administrative efficiency", "military tradition",
            "army tradition", "navy tradition", "innovativeness",
            "colonist", "settler", "native", "discovery", "exploration",
        ],
        notes=(
            "Historical diplomacy and colonial terminology is essential. "
            "Trade mechanics have specific terminology - keep it consistent. "
            "Country names and historical figures should match official translations. "
            "Time period is 1444-1821 - avoid anachronistic terms. "
            "Mission names and event descriptions often have historical flavor. "
            "Estate names and privileges should use consistent terminology."
        ),
        tone="formal",
    ),
    "stellaris": GameProfile(
        game_id="stellaris",
        name="Stellaris",
        genre="Sci-Fi Grand Strategy",
        style="Science fiction, space exploration, futuristic",
        key_terms=[
            "empire", "species", "technology", "fleet", "planet", "star", "system",
            "ship", "corvette", "destroyer", "cruiser", "battleship", "titan", "colossus",
            "energy credits", "minerals", "food", "consumer goods", "alloys", "exotic gases",
            "volatile motes", "rare crystals", "sr_dark_matter", "sr_zro", "sr_living_metal",
            "pop", "faction", "ethics", "authority", "civic", "origin",
            "diplomacy", "federation", "hegemony", "directional", "research",
            "unity", "influence", "amenities", "housing", "crime", "stability",
            "habitability", "district", "building", "megastructure", "ring world",
            "dyson sphere", "matter decompressor", "science ship", "construction ship",
            "terraforming", "ecumenopolis", "machine world", "habitat",
            "ascension", "perks", "tradition", "genetic", "cybernetic", "psionic",
            "synthetic", "machine intelligence", "hive mind", "gestalt consciousness",
        ],
        notes=(
            "Sci-fi terminology should be consistent and futuristic. "
            "Can be more creative and flexible with empire and species names. "
            "Ethics and government types have specific meanings in context. "
            "Technology names should sound scientific and advanced. "
            "Avoid archaic or medieval terms unless describing ancient ruins. "
            "Federation and diplomatic terms should sound modern/futuristic. "
            "Ship class names and component names are technical - translate literally."
        ),
        tone="formal",
    ),
    "vic3": GameProfile(
        game_id="vic3",
        name="Victoria 3",
        genre="Industrial Era Strategy (1836-1936)",
        style="Economic, social, industrial, victorian era",
        key_terms=[
            "population", "pop", "standard of living", "sol", "wealth", "income",
            "employment", "unemployment", "wages", "profit", "tax", "tariff",
            "trade good", "goods", "input", "output", "production method",
            "building", "construction", "infrastructure", "railway", "port",
            "interest group", "lobby", "law", "government", "authority", "legitimacy",
            "radical", "loyalist", "literacy", "innovation", "technology",
            "journal", "event", "decision", "diplomatic play", "war", "peace",
            "recognition", "prestige", "clout", "radicalism", "suppression",
            "capitalist", "landowner", "intelligentsia", "devout", "armed forces",
            "trade union", "petty bourgeoisie", "shopkeepers", "rural folk", "laborers",
            "machine class", "industrialist", "financier", "officer", "clergy",
        ],
        notes=(
            "Victorian-era economic and social terminology is critical. "
            "Interest group names and political movements should be historically accurate. "
            "Economic terms (production, trade, goods) must be consistent. "
            "Law names and government types should match historical usage. "
            "Population types (pops) have specific meanings - translate precisely. "
            "Technology and innovation terms should sound industrial-era. "
            "Diplomatic plays and war mechanics have specific terminology."
        ),
        tone="formal",
    ),
    "imperator": GameProfile(
        game_id="imperator",
        name="Imperator: Rome",
        genre="Ancient Strategy (400 BC - 150 BC)",
        style="Ancient, classical, Hellenistic, Roman Republic era",
        key_terms=[
            "province", "region", "territory", "city", "tribe", "settlement",
            "character", "ruler", "governor", "general", "admiral", "statesman",
            "legion", "cohort", "maniple", "phalanx", "unit", "army", "fleet",
            "gold", "manpower", "provincial trade goods", "trade goods",
            "grain", "wood", "iron", "wine", "olive oil", "textiles", "dyes",
            "copper", "tin", "salt", "horses", "elephants", "camels",
            "stability", "military tradition", "civic tradition", "oratorical power",
            "family", "clan", "house", "dynasty", "marriage", "adoption",
            "religion", "deity", "omen", "religion head", "high priest",
            "republic", "monarchy", "tribal", "dictatorship", "dictator",
            "senate", "consul", "proconsul", "censor", "praetor", "quaestor",
        ],
        notes=(
            "Ancient classical-era terminology is essential. "
            "Greek, Roman, and other ancient civilization terms should be period-accurate. "
            "Character names and titles should match historical usage. "
            "Religious terms should reflect ancient belief systems. "
            "Military unit names should be historically accurate (legion, cohort, etc.). "
            "Trade goods names should match ancient economy. "
            "Government positions should use classical terminology."
        ),
        tone="formal",
    ),
    "generic": GameProfile(
        game_id="generic",
        name="Generic Paradox Game",
        genre="Strategy Game",
        style="General strategy game localization",
        key_terms=[],
        notes=(
            "General strategy game terminology. Use context to determine appropriate translation. "
            "Keep consistent terminology throughout. Preserve game-specific terms as-is if unclear. "
            "When in doubt, translate literally rather than creatively."
        ),
        tone="formal",
    ),
}


def get_game_profile(game_id: str) -> GameProfile:
    """Get game profile by ID. Falls back to generic if not found."""
    return GAME_PROFILES.get(game_id, GAME_PROFILES["generic"])


def get_game_profile_instruction(game_id: str, mod_theme: Optional[str] = None) -> str:
    """Get the instruction text for a specific game profile to include in prompts.
    
    Args:
        game_id: Game identifier (e.g., 'hoi4', 'ck3')
        mod_theme: Optional custom theme description for mods
    
    Returns:
        Instruction string for AI prompt
    """
    profile = get_game_profile(game_id)
    
    # If mod theme is provided, it overrides game-specific instructions
    if mod_theme:
        return (
            f"This is a mod for **{profile.name}** with a custom theme: **{mod_theme}**.\n"
            f"Use vocabulary and style appropriate for the '{mod_theme}' theme. "
            f"Keep all game mechanics terms (like $VARS$, [macros]) intact. "
            f"If the theme conflicts with standard game terminology, prioritize the mod theme. "
            f"Be creative and adapt the language to fit the theme."
        )
    
    if game_id == "generic":
        return ""
    
    terms_text = ""
    if profile.key_terms:
        # Show sample terms (up to 15)
        sample_terms = profile.key_terms[:15]
        terms_text = (
            f"\nKey terminology to be aware of: {', '.join(sample_terms)}, etc. "
            f"Use appropriate translations for these terms that match the game's established vocabulary."
        )
    
    return (
        f"This is a translation for **{profile.name}** ({profile.genre}). "
        f"Use {profile.style} terminology and tone. "
        f"{profile.notes}{terms_text}"
    )


def get_game_list() -> List[tuple]:
    """Get list of all games as (game_id, display_name) tuples."""
    return [(profile.game_id, profile.name) for profile in GAME_PROFILES.values()]
