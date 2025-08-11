# Energiedashboard voor Laadprofielen

Dit Streamlit dashboard analyseert laadprofielen, batterij state-of-charge (SOC) en netcapaciteit.

## Bestanden

- `dashboard.py` - Hoofdapplicatie
- `gridpowertarget.csv` - Beschikbare netcapaciteit per kwartier
- `r_580_3.csv` - SAM simulatie resultaten met energiestromen en SOC
- `requirements.txt` - Python dependencies

## Features

- **Grafiek 1**: Gestapelde energiestromen naar load + Battery SOC
- **Grafiek 2**: Netcapaciteit vs inkoop + ruimte op aansluiting + Battery SOC
- **Sankey diagram**: Energiestromen percentages
- **Export functie**: Download data als CSV
- **Interactieve zoom**: Standaard gefocust op eerste week van juni

## Data

- Batterij SOC in percentage (0-100%)
- Grid power: negatief = inkoop, positief = verkoop
- Tijdreeks begint 1 januari 2025
- 15-minuten resolutie
