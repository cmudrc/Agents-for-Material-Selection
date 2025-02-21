import re
import pandas as pd
import re
from collections import defaultdict

##########################################################################################

searches_df = pd.read_csv('Search Logs Data/searches_by_size.csv')

materials = ['steel', 'aluminum', 'titanium', 'glass', 'wood', 'thermoplastic', 'thermoset', 'elastomer', 'composite']
designs = ['kitchen utensil grip', 'safety helmet', 'underwater component', 'spacecraft component']

# Define criterion-related patterns
criterion_patterns = {
    "heat resistance": r"^(?!necessary).*heat resistance(\S+[\S\s]*)?",
    "heat resistant": r"^(?!necessary).*eat resistant(\S+[\S\s]*)?",
    "material heat resistant": r"^(?!necessary).*(\S+[\S\s]*) heat resistant",
    "melting point": r"^melting point of (\S+[\S\s]*)",
    "heat conductivity of": r"^heat conductivity of (\S+[\S\s]*)",
    "heat resistance scale of": r"^heat resistance scale of (\S+[\S\s]*)",
    "thermal properties of": r"^thermal properties of (\S+[\S\s]*)",
    "corrosion resistance": r"^(?!necessary).*corrosion resistance(\S+[\S\s]*)?",
    "corrosion resistant": r"^(?!necessary).*corrosion resistant(\S+[\S\s]*)?",
    "corrosion": r"^(?!necessary).*(\S+[\S\s]*) corrosion",
    "high strength": r"^high strength (\S+[\S\s]*)",
    "strength of": r"^strength of (\S+[\S\s]*)",
    "strength": r"^^(?!necessary).*(\S+[\S\s]*) strength",
    "strength properties of": r"^strength properties of (\S+[\S\s]*)",
    "lightweight": r"^lightweight (\S+[\S\s]*)",
    "weight of": r"^weight of (\S+[\S\s]*)",
    "weight": r"^(\S+[\S\s]*) weight",
    "density of": r"^density of (\S+[\S\s]*)"
}

# Define use-related patterns
use_patterns = {
    "use of material in design": r"^use of (\S+[\S\s]*) in (\S+[\S\s]*)$",
    "material use in design": r"^(\S+[\S\s]*) use in (\S+[\S\s]*)$",
    "material in design": r"^(\S+[\S\s]*) in (\S+[\S\s]*)$",
    "material for design": r"^(?!necessary properties of)(\S+[\S\s]*) materials for (\S+[\S\s]*)$",
    "underwater end": r"^(\S+[\S\s]*) underwater$",
    "underwater start": r"^underwater (\S+[\S\s]*)",
    "application": r"^(\S+[\S\s]*) application$"
}

material_properties_patterns = {
    "materal properties of": r"^material properties of (" + "|".join([re.escape(material) for material in materials]) + r"s?|(\S+[\S\s]*)s?)$",
    "properties of": r"^properties of (" + "|".join([re.escape(material) for material in materials]) + r"s?|(\S+[\S\s]*)s?)$",
    "other properties of": r"other properties of (" + "|".join([re.escape(material) for material in materials]) + r"s?|(\S+[\S\s]*)s?)$",
    "properties": r"^(" + "|".join([re.escape(material) for material in materials]) + r")\s+properties$"
}

design_properties_patterns = {
    "properties of": r"^properties of (" + "|".join([re.escape(design) for design in designs]) + r")",
    "properties": r"^(" + "|".join([re.escape(design) for design in designs]) + r") properties"
}

requirements_patterns = {
    "necessary": r"^necessary",
    "necesssary": r"^necesssary",
    "requirements": r"^(\S+[\S\s]*) requirements",
    "requirements for": r"^requirements for (\S+[\S\s]*)"
}

performance_patterns = {
    "performance of material in design": r"^performance of (\S+[\S\s]*) in (\S+[\S\s]*)",
    "suitability of material for design": r"^suitability of (\S+[\S\s]*) for (\S+[\S\s]*)",
    "material suitability for design": r"^(\S+[\S\s]*) suitability for (\S+[\S\s]*)",
    "performance": r"^(\S+[\S\s]*) performance",
    "how well material performs as design": r"^how well (\S+[\S\s]*) performs as (\S+[\S\s]*)"
}

# Define main patterns
patterns = {
    "materials": r"^(" + "|".join([re.escape(material) for material in materials]) + r")\s+materials$",
    "material properties": "|".join(material_properties_patterns.values()),
    "design properties": "|".join(design_properties_patterns.values()),
    "usage": "|".join(use_patterns.values()),
    "requirements": "|".join(requirements_patterns.values()),
    "performance": "|".join(performance_patterns.values()),
    "criterion": "|".join(criterion_patterns.values())
}

for query in searches_df["Query"].dropna():
    found = False
    for typology, pattern in patterns.items():
        match = re.match(pattern, query)
        if match:
            if found: print(query)
            found = True