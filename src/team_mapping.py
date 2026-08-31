"""
team_mapping.py
football-data.org (the live fixtures API) and football-data.co.uk (our
historical training data) name teams differently:
  football-data.org: "Manchester United FC"
  football-data.co.uk (what our model was trained on): "Man United"

This maps between them for the 2026-27 Premier League clubs. If a
promoted/relegated club changes next season, this dict is the one place
you'll need to update.

IMPORTANT HONESTY NOTE: I could not test this mapping against a live API
response (my dev environment can't reach football-data.org). The exact
strings the API returns for football-data.org "name" field are my best
knowledge, not a verified fetch. print_unmapped_teams() below is there
specifically to catch any mismatch on your first real run -- if a team
doesn't map, it will print loudly instead of failing silently.
"""

# football-data.org full name -> football-data.co.uk short name (matches data/raw CSVs)
API_TO_CSV_NAME = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
}


def normalize_team_name(api_name: str) -> str:
    """Convert a football-data.org team name to our training-data name."""
    if api_name in API_TO_CSV_NAME:
        return API_TO_CSV_NAME[api_name]

    # Fallback: strip common suffixes and hope for a loose match.
    # This won't be perfect -- it's a safety net, not a substitute for
    # keeping the dict above correct.
    stripped = (
        api_name.replace(" FC", "")
        .replace(" AFC", "")
        .replace("AFC ", "")
    )
    print(f"WARNING: '{api_name}' not in API_TO_CSV_NAME map. "
          f"Guessing '{stripped}' -- verify this matches a team name "
          f"in your training data, or add it to the mapping dict.")
    return stripped


def print_unmapped_teams(api_team_names) -> list:
    """Check a list of API team names against the map; report any misses."""
    unmapped = [t for t in api_team_names if t not in API_TO_CSV_NAME]
    if unmapped:
        print(f"\n{'!' * 60}")
        print(f"{len(unmapped)} team name(s) from the API are NOT in the mapping:")
        for t in unmapped:
            print(f"  - {t}")
        print("Add these to API_TO_CSV_NAME in team_mapping.py before trusting predictions for these teams.")
        print(f"{'!' * 60}\n")
    return unmapped
